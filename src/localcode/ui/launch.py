"""Launch the default localcode interface.

  free ports  →  supervisor (owns llama-server, serves the picker API)
  →  a config file in the run dir points the runtime at that server and loads
     the LocalCode plugin directory  →  run the bundled UI binary in the project.

Nothing is written into the user's project. The runtime's own XDG dirs are
~/.*/localcode-agent. Network use is limited to features the user asks for.
"""
from __future__ import annotations

import json
import os
import sqlite3
import signal
import subprocess
import sys
import time
import urllib.request
from localcode.ui.http import bypass_local_proxies, local_urlopen
from pathlib import Path
from urllib.parse import quote

from localcode.ui import plugin_path, run_dir, ui_binary_path
from localcode.ui.auth import KEY_ENV, TOKEN_ENV, control_headers, new_secret, read_auth_file, server_key
from localcode.ui.ports import choose_ports, find_running

WAIT_S = 240


def _shutdown_supervisor(process) -> None:
    """Allow the guarded model shutdown and kill/reap before forcing exit."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=35)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _model_alias(model: str | None, directory: Path) -> str | None:
    if not model:
        return None
    from localcode.models_catalog import by_key
    from localcode.model_files import find_model
    raw = model.strip()
    choice = by_key(raw)
    filename = choice.filename if choice else Path(raw).name
    if not filename.endswith('.gguf'):
        filename += '.gguf'
    explicit = Path(raw).expanduser()
    existing = explicit if explicit.is_file() else find_model(filename, directory)
    if existing is None:
        return None
    target = directory / existing.name
    directory.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        target.symlink_to(existing.resolve())
    return existing.stem


def _models_dir() -> Path:
    """Where GGUFs live: LOCALCODE_MODEL_DIR (the documented override, also
    honoured by config), LOCALCODE_MODELS_DIR (the older launcher name), else
    the configured/default directory."""
    from localcode.models_catalog import model_dir
    env = os.environ.get("LOCALCODE_MODEL_DIR") or os.environ.get("LOCALCODE_MODELS_DIR")
    return Path(env).expanduser() if env else model_dir()


def _llama_server() -> Path | None:
    env = os.environ.get("LOCALCODE_LLAMA_SERVER")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    from localcode.bootstrap import _turboquant_binary_path
    return _turboquant_binary_path()


def _get_json(url: str, timeout: float = 1.0) -> dict | None:
    """GET a control-API route with the session token."""
    try:
        req = urllib.request.Request(url, headers=control_headers())
        with local_urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def _ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, headers=control_headers())
        with local_urlopen(req, timeout=1.0):
            return True
    except Exception:  # noqa: BLE001
        return False


def _context_size(models_dir: Path, ctrl: int) -> int:
    status = _get_json(f"http://127.0.0.1:{ctrl}/status")
    try:
        ctx = int((status or {}).get("ctx") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx:
        return ctx
    try:
        from localcode.ui.server_cmd import context_size
        return context_size(str(models_dir / "x.gguf"))
    except Exception:  # noqa: BLE001
        # No server and no runtime gateway to ask: the same policy knob the
        # gateway starts from, never a literal token count.
        try:
            from localcode.config import load_config
            return max(2048, int(load_config().max_context_chars) // 4)
        except Exception:  # noqa: BLE001
            return 2048


def _external_directory_rules() -> list[dict]:
    """Ask before touching anything outside the project, as opencode v1 did.

    Upstream v2 stopped asking by default; a local agent writing to arbitrary
    paths without a prompt is not acceptable here. The runtime's own output,
    temp and config folders stay allowed so saved tool output can be read back.
    """
    import tempfile

    home = Path.home()
    data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "localcode-agent"
    config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "localcode-agent"
    tmp = Path(tempfile.gettempdir()) / "localcode-agent"
    allowed = [data / "shell" / "*" / "*", data / "tool-output" / "*", tmp / "*", config / "*"]
    # The runtime compares canonical paths (/var -> /private/var on macOS).
    allowed += [Path(os.path.realpath(p.parent)) / p.name for p in allowed]
    return [{"action": "external_directory", "resource": "*", "effect": "ask"}] + [
        {"action": "external_directory", "resource": str(p), "effect": "allow"} for p in dict.fromkeys(allowed)
    ]


def write_config(path: Path, *, port: int, ctx: int, alias: str | None) -> None:
    """The runtime config for this session, in the runtime's own (v2) schema.
    Provider = the local server only; capabilities are explicit so no cloud
    default leaks in. With no model yet a hidden template entry carries the
    wiring; the runtime's LocalCode provider synthesizes the real alias from it
    once a model is loaded."""
    model_key = alias or "__pending__"
    cfg: dict = {
        "$schema": "https://localcode.dev/schema/config.json",
        "providers": {
            "localcode": {
                "name": "localcode",
                "package": "aisdk:@ai-sdk/openai-compatible",
                "settings": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": server_key() or "local"},
                "models": {
                    model_key: {
                        "name": alias or "no model loaded",
                        "capabilities": {"tools": True, "input": ["text"], "output": ["text"], "temperature": True},
                        # Compaction starts at the input limit (no buffer is configured):
                        # leave room for one response, as opencode v1 did. Once the
                        # supervisor reports its measured budget, the runtime uses that.
                        "limit": {"context": ctx, "output": 8192, **({"input": ctx - 8192} if ctx > 8192 else {})},
                    }
                },
            }
        },
        # Only the local provider may be used.
        "policies": [
            {"action": "provider.use", "resource": "*", "effect": "deny"},
            {"action": "provider.use", "resource": "localcode", "effect": "allow"},
        ],
        # The supervisor's /status budget already includes the headroom one step
        # can add (ui/context_budget.step_reserve), and the provider sets it as
        # the model's input limit; a second reserve here would double-count.
        "compaction": {"buffer": 0},
        # The runtime saves the complete result to disk and exposes its path.
        # Keep the preview affordable, including on small context windows.
        "tool_output": {"max_bytes": max(4096, min(16384, ctx)), "max_lines": 400},
        # No subagents, and no code mode: the model calls each tool directly
        # (code mode's catalog and execute tool would cost every request).
        "permissions": [
            {"action": "subagent", "resource": "*", "effect": "deny"},
            {"action": "execute", "resource": "*", "effect": "deny"},
            *_external_directory_rules(),
        ],
        "agents": {"plan": {"disabled": True}},
        # The plugin directory: server.ts runs in the agent, tui.ts in the terminal UI.
        "plugins": [str(plugin_path())],
        "share": "disabled",
        "update": "disable",
    }
    if alias:
        cfg["model"] = {"providerID": "localcode", "model": alias}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, path)


def _resume_args(resume: str | None) -> list[str]:
    if not resume:
        return []
    return ["--continue"] if resume == "last" else ["--session", resume]


def _session_directory(session_id: str) -> Path | None:
    """Find the project recorded for a session in the local runtime database."""
    if not session_id.startswith("ses_"):
        return None
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    for db in (data_home / "localcode-agent" / "opencode.db",
               data_home / "localcode-agent" / "opencode-localcode.db"):
        if not db.is_file():
            continue
        # session_v2 is the current runtime's table; `session` is an older
        # database the runtime has not opened (and migrated) yet.
        for table in ("session_v2", "session"):
            try:
                with sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True) as conn:
                    row = conn.execute(f"SELECT directory FROM {table} WHERE id = ?", (session_id,)).fetchone()
            except sqlite3.Error:
                continue
            if row and row[0]:
                directory = Path(row[0]).expanduser()
                if directory.is_dir():
                    return directory.resolve()
    return None


def runtime_env(config_path: Path, ctrl: int, *, trust_project: bool = False) -> dict[str, str]:
    """Project configuration can execute code before tool permissions apply."""
    env = dict(os.environ)
    env.setdefault("LOCALCODE_CONTEXT", "minimal")
    bypass_local_proxies(env)
    env["LOCALCODE_CONTROL_URL"] = f"http://127.0.0.1:{ctrl}"
    env["LOCALCODE_CONFIG"] = str(config_path)
    # The branded alias takes precedence in the fork. Set both explicitly so
    # an inherited alias cannot accidentally reopen project code loading.
    disabled = "0" if trust_project else "1"
    env["LOCALCODE_DISABLE_PROJECT_CONFIG"] = disabled
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = disabled
    # Claude Code's skill directories (~/.claude/skills, <project>/.claude/skills)
    # are written for Claude and listed in full in every request's system
    # prompt: one install put 63 of them (~12k tokens) in front of a 12B
    # model on every session. Local models keep their own skill dirs
    # (.localcode-agent/skills, ~/.agents/skills, config `skills.paths`).
    # Set OPENCODE_DISABLE_CLAUDE_CODE_SKILLS=0 to opt back in.
    env.setdefault("OPENCODE_DISABLE_CLAUDE_CODE_SKILLS", "1")
    return env


MCP_HELP = """usage: localcode mcp <add|list|auth|logout> [args] [--trust-project]

Manage the MCP servers the agent can use. The commands are the runtime's own:
  add       localcode mcp add <name> -- <command> [args...]   (a local server)
            localcode mcp add <name> --url <url>              (a remote server)
            writes the server into this project's localcode.json; add --global
            to write it into your user config so it loads in every project
  list      servers and their status (OAuth state for remote servers)
  auth      sign in to a remote server that uses OAuth: localcode mcp auth <name>
  logout    forget a remote server's OAuth tokens

Servers declared in the project's localcode.json load only when you launch
with --trust-project (project configuration can run code). Pass the same flag
here to see them in `list`. Servers in ~/.config/localcode-agent/localcode.json
always load. In a session, /mcps toggles servers on and off.
"""

PROJECT_CONFIGS = (
    "localcode.json", "localcode.jsonc", ".localcode-agent/localcode.json", ".localcode-agent/localcode.jsonc",
    "opencode.json", "opencode.jsonc", ".opencode/opencode.json", ".opencode/opencode.jsonc",
)


def _mcp_server_names(mcp: object) -> list[str]:
    """Server names in an `mcp` block: `mcp.servers` (current) or flat entries (legacy)."""
    if not isinstance(mcp, dict):
        return []
    names = set(mcp["servers"]) if isinstance(mcp.get("servers"), dict) else set()
    names.update(name for name, value in mcp.items()
                 if name not in {"servers", "timeout"} and isinstance(value, dict))
    return sorted(names)


def _project_mcp_names(project_dir: Path) -> list[str]:
    """Server names declared in the project's own config files, if any."""
    names: set[str] = set()
    for name in PROJECT_CONFIGS:
        f = project_dir / name
        if not f.is_file():
            continue
        try:
            # .jsonc: drop whole-line // comments only; a "//" inside a URL stays.
            text = "\n".join(line for line in f.read_text().splitlines() if not line.lstrip().startswith("//"))
            data = json.loads(text)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            names.update(_mcp_server_names(data.get("mcp")))
    return sorted(names)


def mcp_main(argv: list[str]) -> int:
    """`localcode mcp ...`: run the runtime's own MCP commands with this
    install's config and trust environment, so a user never edits JSON by hand
    to add a server."""
    args = list(argv)
    trust = "--trust-project" in args
    args = [a for a in args if a != "--trust-project"]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(MCP_HELP, end="")
        return 0
    ui_bin = ui_binary_path()
    if ui_bin is None:
        print("localcode: the UI binary is missing from this install. Reinstall with `pip install -U localcode`.", file=sys.stderr)
        return 1
    project_dir = Path.cwd()
    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    config_path = rd / f"mcp-{os.getpid()}.json"
    # No model is involved; the session config only carries the provider wiring
    # the runtime expects to find, plus the plugin, and is removed afterwards.
    write_config(config_path, port=0, ctx=0, alias=None)
    env = runtime_env(config_path, 0, trust_project=trust)
    before = set(_project_mcp_names(project_dir))
    try:
        code = subprocess.call([str(ui_bin), "mcp", *args], cwd=str(project_dir), env=env)
    finally:
        try:
            config_path.unlink()
        except OSError:
            pass
    after = _project_mcp_names(project_dir)
    added = [n for n in after if n not in before]
    if args[0] == "add" and added:
        print(f"localcode: {', '.join(added)} is declared in this project's localcode.json. "
              "Project servers load only when you launch with `localcode --trust-project`; "
              "for every project, add it to your user config instead (localcode mcp add --global ...).",
              file=sys.stderr)
    elif args[0] == "list" and after and not trust:
        print(f"localcode: this project also declares {', '.join(after)}; they are hidden here and in sessions "
              "until you pass --trust-project.", file=sys.stderr)
    return code


def attach(ctrl: int, status: dict, ui_bin: Path, project_dir: Path, alias: str | None,
           resume: str | None = None, *, trust_project: bool = False) -> int:
    """Open another UI window on the model server a running session owns."""
    # The running session's secrets, left owner-only in the run directory.
    auth = read_auth_file(run_dir())
    if auth:
        os.environ[TOKEN_ENV], os.environ[KEY_ENV] = auth
    port = int(status.get("inference_port") or status["port"])
    try:
        ctx = int(status.get("ctx") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if not ctx:
        ctx = _context_size(_models_dir(), ctrl)
    current = status.get("current") or None
    if alias and current and alias != current:
        print(f"localcode: attaching to the running session's model {current}; use /models there to switch.",
              file=sys.stderr)
    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    config_path = rd / f"session-{os.getpid()}.json"
    write_config(config_path, port=port, ctx=ctx, alias=current)
    env = runtime_env(config_path, ctrl, trust_project=trust_project)
    env["PWD"] = str(project_dir)   # the runtime's non-TUI modes take their directory from PWD
    argv = [str(ui_bin)] + (["-m", f"localcode/{current}"] if current else []) + _resume_args(resume)
    try:
        return subprocess.call(argv, cwd=str(project_dir), env=env)
    finally:
        try:
            config_path.unlink()
        except OSError:
            pass


def main(model: str | None = None, project: str | None = None,
         resume: str | None = None, *, trust_project: bool = False) -> int:
    ui_bin = ui_binary_path()
    if ui_bin is None:
        print("localcode: the UI binary is missing from this install. "
              "Reinstall with `pip install -U localcode`.",
              file=sys.stderr)
        return 1
    server = _llama_server()
    if server is None:
        print("localcode: bundled llama-server not found (broken install). Reinstall with `pip install -U localcode`.",
              file=sys.stderr)
        return 1
    models_dir = _models_dir()
    try:
        alias = _model_alias(model, models_dir)
    except OSError as error:
        print(f"localcode: could not use the existing model: {error}", file=sys.stderr)
        return 1
    if model and alias is None:
        print(f"localcode: no downloaded model named {model!r} in the model folders; opening the picker instead.",
              file=sys.stderr)
    project_dir = Path(project).expanduser().resolve() if project else (
        _session_directory(resume) if resume and resume != "last" else None
    ) or Path.cwd()

    # A session already open in another terminal: attach to its model server
    # instead of refusing. One llama-server per machine, any number of windows.
    running = find_running()
    if running is not None:
        from localcode import __version__
        if running[1].get("version") != __version__:
            print(f"localcode: a service from an earlier installation is still running. "
                  f"Exit LocalCode in its open terminals, then launch again to use {__version__}.",
                  file=sys.stderr)
            return 1
        return attach(running[0], running[1], ui_bin, project_dir, alias, resume,
                      trust_project=trust_project)

    try:
        port, ctrl = choose_ports()
    except RuntimeError as error:
        print(f"localcode: {error}", file=sys.stderr)
        return 1

    # Fresh secrets for this session; the supervisor and the UI inherit them.
    os.environ[TOKEN_ENV] = new_secret()
    os.environ[KEY_ENV] = new_secret()

    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(rd, 0o700)   # server.log, warm-up token files and auth.json live here
    except OSError:
        pass
    sup_log = (rd / "supervisor.log").open("ab")
    sup = subprocess.Popen(
        [sys.executable, "-m", "localcode.ui.supervisor",
         "--model", alias or "", "--port", str(port), "--control-port", str(ctrl),
         "--server", str(server), "--models-dir", str(models_dir), "--parent-pid", str(os.getpid())],
        stdout=sup_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    frontend: subprocess.Popen | None = None

    def cleanup() -> None:
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
        _shutdown_supervisor(sup)

    def on_signal(signum, _frame):
        cleanup()
        raise SystemExit(128 + signum)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    try:
        ready_url = f"http://127.0.0.1:{port}/health" if alias else f"http://127.0.0.1:{ctrl}/status"
        deadline = time.monotonic() + WAIT_S
        while not _ok(ready_url):
            if sup.poll() is not None:
                print(f"localcode: the model server failed to start (see {rd / 'supervisor.log'})", file=sys.stderr)
                return 1
            if time.monotonic() > deadline:
                print("localcode: timed out waiting for the model server", file=sys.stderr)
                return 1
            time.sleep(0.5)

        status = _get_json(f"http://127.0.0.1:{ctrl}/status", timeout=2)
        if not status or not status.get("inference_port"):
            print("localcode: could not discover the model endpoint; relaunch LocalCode", file=sys.stderr)
            return 1
        ctx = _context_size(models_dir, ctrl)
        config_path = rd / "session.json"
        write_config(config_path, port=int(status["inference_port"]), ctx=ctx, alias=alias)

        env = runtime_env(config_path, ctrl, trust_project=trust_project)
        env["PWD"] = str(project_dir)   # the runtime's non-TUI modes take their directory from PWD
        argv = [str(ui_bin)]
        if alias:
            argv += ["-m", f"localcode/{alias}"]
        argv += _resume_args(resume)
        frontend = subprocess.Popen(argv, cwd=str(project_dir), env=env)
        # Watch both: if the supervisor dies (killed, crashed), the UI would
        # sit on "No model loaded" with a picker that never answers. End the
        # session with a clear message instead.
        while True:
            rc = frontend.poll()
            if rc is not None:
                return rc
            if sup.poll() is not None:
                frontend.terminate()
                try:
                    frontend.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    frontend.kill()
                print(f"localcode: the model service stopped unexpectedly (see {rd / 'supervisor.log'}). "
                      "Run `localcode` again.", file=sys.stderr)
                return 1
            time.sleep(0.5)
    finally:
        cleanup()
        sup_log.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None))
