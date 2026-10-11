"""The agent under test, as Inspect AI components.

`localcode/<alias>` is an Inspect model provider that names an installed GGUF.
`localcode_ui()` is the solver: it runs the real bundled UI headlessly
(`localcode-ui run --format json`) against the real bundled llama-server, one
turn per `## Turn N` section of the task's instruction, inside the sample's
sandbox directory, and records what happened as the sample transcript.

Inspect does not sit between the UI and the server. The UI talks to
llama-server exactly as it does for a user, because prompt caching, slots and
streaming are part of what is measured; a proxy in the middle would change them.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import anyio
from inspect_ai.model import (ChatMessageAssistant, ChatMessageTool, ChatMessageUser, GenerateConfig, ModelAPI,
                              ModelOutput, ModelUsage, modelapi)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_ai.util import concurrency, sandbox

from .performance import ResourceMeter, summarize
from .process import run_bounded

EVALS = Path(__file__).resolve().parent.parent
REPO = Path(os.environ.get("LOCALCODE_EVALS_REPO") or EVALS.parent).resolve()
UI = REPO / "src/localcode/bin/localcode-ui"
SERVER = REPO / "src/localcode/bin/llama-server"
MODELS_DIR = Path.home() / ".local/share/localcode/models"

# A tool call written out as prose instead of emitted: JSON with a tool name and
# arguments, or the XML-ish wrappers some chat templates use.
NARRATED_CALL_RE = re.compile(
    r'<(?:tool_call|function_call|invoke|function=)[\s>]|"(?:name|tool_name|function)"\s*:\s*"[a-z_]+"\s*,\s*"(?:arguments|parameters|input)"\s*:', re.I)
CACHE_NOISE_RE = re.compile(r"(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/)|\.pyc$")


def _python() -> str:
    if os.environ.get("LOCALCODE_PY"):
        return os.environ["LOCALCODE_PY"]
    venv = REPO / ".venv/bin/python"
    return str(venv) if venv.exists() else sys.executable


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


class Runtime:
    """One isolated supervisor and llama-server: throwaway HOME, spare ports,
    single slot. Only model files are shared with the real home, read-only."""

    def __init__(self, model: str, ram_gb: int | None, env: dict[str, str]):
        self.requested, self.ram_gb, self.overrides = model, ram_gb, dict(env)
        self.proc = None; self.log = None
        self._prepare()
        alias = model
        if Path(model).name != model:
            raise RuntimeError(f"{model!r} is not a model name")
        if not (MODELS_DIR / f"{model}.gguf").is_file():
            # A catalog group key: ask the picker which quant this machine (or the
            # emulated one) gets, exactly as the first-run screen would.
            alias = self._picked(model)
            if not (MODELS_DIR / f"{alias}.gguf").is_file():
                self.stop(); raise RuntimeError(f"the picker chose {alias} for {model}, which is not installed (nothing is downloaded for you)")
        self.alias = alias
        self._start(alias)
        try:
            self._wait_ready()
        except BaseException:
            self.stop(keep_home=True)
            raise
        self.cfg = self.home / "session.json"
        subprocess.run([_python(), "-c",
                        "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from localcode.ui.launch import write_config; "
                        "write_config(Path(sys.argv[2]), port=int(sys.argv[3]), ctx=int(sys.argv[4]), alias=sys.argv[5])",
                        str(REPO / "src"), str(self.cfg), str(self.mport), str(int(self.status.get("ctx") or 32768)), alias],
                       env=self.base_env, check=True)
        self.binary_sha = subprocess.run(["shasum", "-a", "256", str(UI)], capture_output=True, text=True).stdout[:12]

    def _prepare(self) -> None:
        ram_gb, env = self.ram_gb, self.overrides
        self.home = Path(tempfile.mkdtemp(prefix="localcode-evals-home-")).resolve()
        self.run_dir = self.home / "run"; self.run_dir.mkdir()
        self.links = links = self.home / ".local/share/localcode/models"; links.mkdir(parents=True)
        for f in MODELS_DIR.glob("*.gguf"):
            if f.is_file(): (links / f.name).symlink_to(f)
        # Browser builds already on this machine are shared the same way; none is fetched.
        browsers = Path.home() / "Library/Caches/ms-playwright"
        if browsers.is_dir():
            (self.home / "Library/Caches").mkdir(parents=True)
            (self.home / "Library/Caches/ms-playwright").symlink_to(browsers)
        self.token, self.key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.base_env = {
            **os.environ, **self.overrides,
            "HOME": str(self.home), "XDG_CONFIG_HOME": str(self.home / ".config"), "XDG_DATA_HOME": str(self.home / ".local/share"),
            "XDG_CACHE_HOME": str(self.home / ".cache"), "XDG_STATE_HOME": str(self.home / ".local/state"),
            "LOCALCODE_CONTROL_TOKEN": self.token, "LOCALCODE_SERVER_KEY": self.key,
            "LOCALCODE_AGENT_RUN_DIR": str(self.run_dir), "LOCALCODE_PARALLEL": "1",
        }
        if ram_gb: self.base_env["LOCALCODE_RAM_GB"] = str(ram_gb)
        else: self.base_env.pop("LOCALCODE_RAM_GB", None)

    def _start(self, alias: str) -> None:
        self.mport, self.cport = _free_port(), _free_port()
        self.log = open(self.home / "supervisor.log", "ab")
        self.proc = subprocess.Popen(
            [_python(), "-m", "localcode.ui.supervisor", "--model", alias, "--port", str(self.mport), "--control-port", str(self.cport),
             "--server", str(SERVER), "--models-dir", str(self.links), "--parent-pid", str(os.getpid())],
            env={**self.base_env, "PYTHONPATH": str(REPO / "src")}, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def _halt(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=15)
            except Exception: self.proc.kill()
        if self.log is not None: self.log.close()
        self.proc = None; self.log = None

    def _picked(self, group: str) -> str:
        self._start("")
        try:
            for _ in range(100):
                try: self.control("/status"); break
                except Exception: time.sleep(0.3)
            else:
                raise RuntimeError(f"supervisor did not start; see {self.home / 'supervisor.log'}")
            quants = self.control(f"/quants?group={group}", timeout=60).get("quants") or []
        finally:
            self._halt()
        chosen = [q for q in quants if q.get("recommended")]
        if not chosen:
            self.stop(); raise RuntimeError(f"{group!r} is neither an installed GGUF nor a catalog group with a quant that fits")
        return chosen[0]["alias"]

    def control(self, path: str, timeout: float = 2) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.cport}{path}", headers={"X-Localcode-Token": self.token})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _wait_ready(self) -> None:
        for _ in range(100):
            try: self.control("/status"); break
            except Exception: time.sleep(0.3)
        else:
            raise RuntimeError(f"supervisor did not start; see {self.home / 'supervisor.log'}")
        for _ in range(900):
            if self.proc.poll() is not None:
                raise RuntimeError(f"supervisor exited; see {self.home / 'supervisor.log'}")
            self.status = self.control("/status")
            if self.status.get("state") == "ready" and self.status.get("current") == self.alias:
                time.sleep(3)
                return
            if self.status.get("state") == "error":
                raise RuntimeError(f"model failed to load: {self.status}")
            time.sleep(1)
        raise RuntimeError("model did not become ready")

    def describe(self) -> dict:
        """What the picker gave this model on this (possibly emulated) machine."""
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.mport}/props", headers={"Authorization": "Bearer " + self.key})
            with urllib.request.urlopen(req, timeout=5) as r: props = json.load(r)
        except (OSError, ValueError):
            props = {}
        return {"model": self.alias, "requested": self.requested, "context": self.status.get("ctx"), "slots": props.get("total_slots"),
                "generation_context": (props.get("default_generation_settings") or {}).get("n_ctx"),
                "ui_binary_sha": self.binary_sha, "ram_gb": self.ram_gb, "env": self.overrides}

    def server_log(self) -> str:
        f = self.run_dir / "server.log"
        return f.read_text(errors="replace") if f.exists() else ""

    def stop(self, keep_home: bool = False) -> None:
        self._halt()
        if not keep_home and not os.environ.get("LOCALCODE_EVALS_KEEP_HOME"):
            shutil.rmtree(self.home, ignore_errors=True)


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def runtime_for(alias: str, ram_gb: int | None, env: dict[str, str]) -> Runtime:
    """The loaded runtime, replaced when a different model or setting is asked
    for. One model in memory at a time."""
    global _runtime
    with _runtime_lock:
        want = (alias, ram_gb, tuple(sorted(env.items())))
        if _runtime is not None and (_runtime.requested, _runtime.ram_gb, tuple(sorted(_runtime.overrides.items()))) != want:
            _runtime.stop(); _runtime = None
        if _runtime is None:
            _runtime = Runtime(alias, ram_gb, env)
        return _runtime


def stop_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.stop(); _runtime = None


atexit.register(stop_runtime)


@modelapi(name="localcode")
def _localcode_provider():
    class LocalCodeAPI(ModelAPI):
        """Names an installed GGUF (`--model localcode/<alias>`). The supervisor
        that serves it is started by the solver, so several models can be listed
        in one `inspect eval-set` without all of them loading at once."""

        def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                     config: GenerateConfig = GenerateConfig(), **_: Any) -> None:
            super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)

        async def generate(self, input, tools, tool_choice, config) -> ModelOutput:  # noqa: A002
            raise NotImplementedError("the agent under test is the LocalCode UI: use the localcode_ui() solver")

        async def aclose(self) -> None:
            stop_runtime()

    return LocalCodeAPI


def turns_of(instruction: str) -> list[str]:
    if "## Turn 1" in instruction:
        return [s.strip() for s in re.split(r"^## Turn \d+\s*$", instruction, flags=re.M) if s.strip()]
    return [instruction.strip()]


def _write(path: Path, text: str) -> None:
    """The .eval/ folder lives in the workspace, and a model told to clean caches
    may delete it mid-run (seen on forbidden-commands). Recreate it on every write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_trial(rt: Runtime, ws: Path, turns: list[str], meta: dict) -> tuple[list, dict]:
    """Drive every turn of one trial in workspace `ws`. Returns the transcript
    and the measurements, and leaves the `.eval/` contract the grader reads."""
    git = lambda *x: subprocess.run(["git", "-c", "user.email=e@x", "-c", "user.name=e", *x], cwd=ws, capture_output=True, text=True)
    git("init", "-q"); git("add", "-A"); git("commit", "-qm", "init", "--allow-empty")
    ev = ws / ".eval"; ev.mkdir(exist_ok=True)
    env = {**rt.base_env, "LOCALCODE_TEST_HOME": str(rt.home), "PWD": str(ws), "LOCALCODE_CONFIG": str(rt.cfg),
           "LOCALCODE_CONTROL_URL": f"http://127.0.0.1:{rt.cport}", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1"}
    # Task-level MCP servers go into the session config for this trial only.
    session = json.loads(rt.cfg.read_text()); session.pop("mcp", None)
    if meta.get("mcp"): session["mcp"] = meta["mcp"]
    rt.cfg.write_text(json.dumps(session, indent=2))
    setup_proc = None
    if meta.get("setup"):
        setup_proc = subprocess.Popen(["bash", "-lc", meta["setup"]], cwd=ws, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1.5)

    prompt_evals = lambda: [int(m) for m in re.findall(r"prompt eval time = +[0-9.]+ ms / +(\d+) tokens", rt.server_log())]
    prompt_lengths = lambda: [int(m) for m in re.findall(r"stop processing: n_tokens = (\d+)", rt.server_log())]
    def budget_now():
        try: return rt.control("/status").get("budget")
        except Exception: return None

    messages: list = []; events: list[dict] = []; commands: list[str] = []
    steps = tool_calls = invalid = rejected = guarded = narrated = 0
    invalid_samples: list[str] = []; calls_per_turn: list[int] = []; reread: list[int] = []; budgets: list = []; first_text: list[float] = []
    final = ""; wall = 0.0; timed_out = runtime_failed = False; stderr_tail = ""
    lengths_before = len(prompt_lengths())
    server_log_path = rt.run_dir / "server.log"
    log_offset = server_log_path.stat().st_size if server_log_path.exists() else 0
    ui_logs = rt.home / ".local/share/localcode-agent/log"
    ui_offsets = {p: p.stat().st_size for p in ui_logs.glob("*.log")}
    meter = ResourceMeter(rt.proc.pid)
    budget = int(meta.get("budget_seconds", 900))
    try:
        for i, text in enumerate(turns, 1):
            messages.append(ChatMessageUser(content=text))
            before = len(prompt_evals())
            args = [str(UI), "run", "--format", "json", "-m", f"localcode/{rt.alias}"] + (["--continue"] if i > 1 else []) + [text]
            t0 = time.time()
            try:
                r = run_bounded(args, cwd=ws, env=env, text=True, timeout=budget - wall, stdin=subprocess.DEVNULL)
                out = r.stdout; runtime_failed = runtime_failed or r.returncode != 0
                _write(ev / f"stderr-turn{i}.txt", r.stderr)
                if r.returncode != 0: stderr_tail = r.stderr[-1500:]
            except subprocess.TimeoutExpired as e:
                out = e.stdout or ""; timed_out = True
                if isinstance(out, bytes): out = out.decode(errors="replace")
            wall += time.time() - t0
            # The client exits on the last streamed token; the server prints that
            # request's timing lines a few ms later. Wait so they land in this turn.
            for _ in range(60):
                txt = rt.server_log()
                if txt.count("launch_slot_: id") <= txt.count("stop processing: n_tokens"): break
                time.sleep(0.05)
            _write(ev / f"events-turn{i}.jsonl", out)
            turn_text: list[str] = []; pending: list[str] = []; turn_calls = 0
            for line in out.splitlines():
                try: e = json.loads(line)
                except Exception: continue
                events.append(e)
                kind = e.get("type"); part = e.get("part") or {}
                if kind == "first_text_delta" and isinstance(e.get("timestamp"), (int, float)):
                    first_text.append(max(0, e["timestamp"] / 1000 - t0))
                elif kind == "step_start":
                    steps += 1
                elif kind == "text":
                    txt = str(part.get("text") or e.get("text") or "")
                    turn_text.append(txt); pending.append(txt)
                    if NARRATED_CALL_RE.search(txt): narrated += 1
                elif kind == "tool_use":
                    tool_calls += 1; turn_calls += 1
                    st = part.get("state") or {}
                    tool = str(part.get("tool") or "tool")
                    cid = str(part.get("callID") or part.get("id") or f"call-{i}-{turn_calls}")
                    failed = st.get("status") == "error"
                    msg = str(st.get("error", "")) if failed else ""
                    if failed:
                        # Refusals caused by the harness (headless permission policy, the
                        # plugin's own guards) are not the model's fault; everything else
                        # is a call the model emitted that could not execute.
                        if "rejected permission" in msg: rejected += 1
                        elif msg.startswith("SYSTEM:") or "LocalCode" in msg or "head, tail" in msg: guarded += 1
                        else: invalid += 1; invalid_samples.append(msg[:160])
                    if tool == "bash": commands.append(str((st.get("input") or {}).get("command", "")))
                    arguments = st.get("input") if isinstance(st.get("input"), dict) else {}
                    messages.append(ChatMessageAssistant(content="\n".join(pending).strip(), tool_calls=[ToolCall(id=cid, function=tool, arguments=arguments)]))
                    pending = []
                    messages.append(ChatMessageTool(content=(msg if failed else str(st.get("output") or ""))[:8000], tool_call_id=cid, function=tool,
                                                    error=ToolCallError(type="unknown", message=msg[:500]) if failed else None))
            if pending or not turn_calls:
                messages.append(ChatMessageAssistant(content="\n".join(pending).strip() or "(no text)"))
            final = "\n".join(turn_text).strip() or final
            calls_per_turn.append(turn_calls)
            _write(ev / f"final_turn{i}.txt", "\n".join(turn_text).strip())
            agent_tokens = [n for n in prompt_evals()[before:] if n > 20]      # drop the tiny title/side requests
            reread.append(agent_tokens[0] if agent_tokens else 0)
            budgets.append(budget_now())
            if timed_out: break
    finally:
        resources = meter.finish()
        if meta.get("teardown"):
            subprocess.run(["bash", "-lc", meta["teardown"]], cwd=ws, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if setup_proc is not None and setup_proc.poll() is None:
            try: os.killpg(setup_proc.pid, signal.SIGTERM)
            except OSError: pass

    # The .eval contract the hidden grader reads (tasks/README.md).
    status = git("status", "--porcelain", "--untracked-files=all").stdout
    changed = sorted({l[3:].split(" -> ")[-1] for l in status.splitlines()
                      if l.strip() and not l[3:].startswith((".eval/", ".localcode-agent/")) and not CACHE_NOISE_RE.search(l[3:])})
    _write(ev / "diff.txt", "\n".join(changed) + ("\n" if changed else ""))
    _write(ev / "final.txt", final)
    _write(ev / "commands.log", "\n".join(commands) + ("\n" if commands else ""))
    ui_text = ""
    for path in ui_logs.glob("*.log"):
        with path.open("rb") as stream:
            stream.seek(ui_offsets.get(path, 0)); ui_text += stream.read().decode(errors="replace")
    ui_errors = [line for line in ui_text.splitlines() if "level=ERROR" in line]
    _write(ev / "ui_errors.txt", "\n".join(ui_errors))
    lengths = prompt_lengths()[lengths_before:]
    with server_log_path.open("rb") as stream:
        stream.seek(log_offset); performance = summarize(events, stream.read().decode(errors="replace"))
    # Tool-call validity: of the calls the model meant to make, how many executed.
    counted = tool_calls + narrated - rejected - guarded
    metrics = {
        "steps": steps, "tool_calls": tool_calls, "tool_calls_per_turn": calls_per_turn, "wall_s": round(wall, 1), "reread_tokens": reread,
        "tool_invalid": invalid, "tool_narrated": narrated, "tool_rejected": rejected, "tool_guarded": guarded,
        "tool_validity": round((counted - invalid - narrated) / counted, 3) if counted > 0 else None,
        "tool_invalid_samples": invalid_samples[:5], "prompt_tokens_max": max(lengths) if lengths else 0,
        "prompt_tokens_by_request": lengths, "budget_by_turn": budgets, **resources, **performance,
        "end_to_end_first_text_s": first_text or None,
        "compactions": sum("agent=compaction" in line and "message=stream" in line for line in ui_text.splitlines()),
        "ui_errors": len(ui_errors), "ui_error_lines": [e[:400] for e in ui_errors[:5]], "stderr_tail": stderr_tail, "timed_out": timed_out, "runtime_failed": runtime_failed,
        "changed_files": changed, "diff": git("diff").stdout[:20000], "final": final,
    }
    _write(ev / "metrics.json", json.dumps({k: v for k, v in metrics.items() if k not in ("diff", "final")}))
    return messages, metrics


def _record_usage(state: TaskState, usage: dict | None) -> ModelUsage | None:
    """Count the server's tokens against the sample so Inspect's totals are real."""
    if not usage:
        return None
    mu = ModelUsage(input_tokens=int(usage.get("input", 0)), output_tokens=int(usage.get("output", 0)),
                    total_tokens=int(usage.get("input", 0)) + int(usage.get("output", 0)),
                    input_tokens_cache_read=int(usage.get("cache_read", 0)) or None,
                    input_tokens_cache_write=int(usage.get("cache_write", 0)) or None)
    try:
        from inspect_ai.model import get_model
        from inspect_ai.model._model import record_and_check_model_usage
        record_and_check_model_usage(get_model(), mu)
    except Exception:  # noqa: BLE001 - bookkeeping only; the numbers are also in the score metadata
        pass
    return mu


@solver
def localcode_ui(ram_gb: int | None = None, env: dict[str, str] | None = None) -> Solver:
    """Run the sample's instruction through the real LocalCode UI.

    ram_gb: emulate a Mac with this much memory (the picker's context and slots).
    env: LocalCode launcher overrides for an A/B run, e.g. {"LOCALCODE_LLAMA_CPP_MTP_DRAFT": "0"}.
    """
    overrides = {str(k): str(v) for k, v in (env or {}).items()}

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.model.api != "localcode":
            raise RuntimeError("run with --model localcode/<installed GGUF name without .gguf>")
        workspace = Path((await sandbox().exec(["pwd", "-P"])).stdout.strip())
        # One model in memory, one trial at a time; time spent queued is not working time.
        async with concurrency("localcode", 1):
            rt = await anyio.to_thread.run_sync(runtime_for, state.model.name, ram_gb, overrides)
            # The last user message, so a setup solver can fill in placeholders first.
            messages, metrics = await anyio.to_thread.run_sync(run_trial, rt, workspace, turns_of(state.user_prompt.text), state.metadata)
            runtime = rt.describe()
        state.messages = messages
        usage = _record_usage(state, metrics.get("usage_tokens"))
        state.output = ModelOutput.from_content(model=str(state.model), content=metrics.pop("final") or "")
        if usage: state.output.usage = usage
        state.store.set("metrics", metrics)
        state.store.set("runtime", runtime)
        return state

    return solve
