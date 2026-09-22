"""Launch the default localcode interface.

  free ports  →  supervisor (owns llama-server, serves the picker API)
  →  a config file in the run dir points the runtime at that server and loads
     the discipline plugin  →  run the bundled UI binary in the project.

Nothing is written into the user's project. The runtime's own XDG dirs are
~/.*/localcode-agent. Network use is limited to features the user asks for.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from localcode.ui import plugin_path, run_dir, ui_binary_path
from localcode.ui.ports import choose_ports

WAIT_S = 240


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
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def _ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0):
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
        return 32768


def write_config(path: Path, *, port: int, ctx: int, alias: str | None) -> None:
    """The runtime config for this session. Provider = the local server only;
    capabilities are explicit so no cloud default leaks in. With no model yet a
    hidden template entry carries the wiring; the picker synthesizes the real
    alias from it once a model is loaded."""
    model_key = alias or "__pending__"
    cfg: dict = {
        "$schema": "https://localcode.dev/schema/config.json",
        "provider": {
            "localcode": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "localcode",
                "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "local"},
                "models": {
                    model_key: {
                        "name": alias or "no model loaded",
                        "tool_call": True, "reasoning": False, "temperature": True, "attachment": False,
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "limit": {"context": ctx, "output": 8192},
                    }
                },
            }
        },
        "enabled_providers": ["localcode"],
        "tools": {"task": False},
        "agent": {"plan": {"disable": True}},
        "lsp": True,
        "plugin": [str(plugin_path())],
        "share": "disabled",
        "autoupdate": False,
    }
    if alias:
        cfg["model"] = f"localcode/{alias}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, path)


def main(model: str | None = None, project: str | None = None) -> int:
    ui_bin = ui_binary_path()
    if ui_bin is None:
        print("localcode: the UI binary is missing from this install. "
              "Run `localcode --classic` for the previous interface, or reinstall with `pip install -U localcode`.",
              file=sys.stderr)
        return 1
    server = _llama_server()
    if server is None:
        print("localcode: bundled llama-server not found (broken install). Reinstall with `pip install -U localcode`.",
              file=sys.stderr)
        return 1
    models_dir = _models_dir()
    alias = (model or "").strip() or None
    if alias and not (models_dir / f"{alias}.gguf").is_file():
        print(f"localcode: no downloaded model named {alias!r} in {models_dir}; opening the picker instead.",
              file=sys.stderr)
        alias = None
    project_dir = Path(project).expanduser().resolve() if project else Path.cwd()
    if project_dir in (Path.home().resolve(), Path("/")):
        # The runtime indexes the working directory as the project; on a home
        # directory that means scanning everything you own before the first
        # answer, and it never finishes in practice.
        print(f"localcode: {project_dir} is your home directory, not a project. "
              "cd into the project you want to work on (or `mkdir myapp && cd myapp`) and run localcode there.",
              file=sys.stderr)
        return 1

    try:
        port, ctrl = choose_ports()
    except RuntimeError as error:
        print(f"localcode: {error}", file=sys.stderr)
        return 1

    rd = run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    sup_log = (rd / "supervisor.log").open("ab")
    sup = subprocess.Popen(
        [sys.executable, "-m", "localcode.ui.supervisor",
         "--model", alias or "", "--port", str(port), "--control-port", str(ctrl),
         "--server", str(server), "--models-dir", str(models_dir), "--parent-pid", str(os.getpid())],
        stdout=sup_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    frontend: subprocess.Popen | None = None

    def cleanup() -> None:
        for proc in (frontend, sup):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        if sup.poll() is None:
            try:
                sup.wait(timeout=8)
            except subprocess.TimeoutExpired:
                sup.kill()

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

        ctx = _context_size(models_dir, ctrl)
        config_path = rd / "session.json"
        write_config(config_path, port=port, ctx=ctx, alias=alias)

        env = dict(os.environ)
        env["LOCALCODE_CONTROL_URL"] = f"http://127.0.0.1:{ctrl}"
        env["LOCALCODE_CONFIG"] = str(config_path)
        # Language servers are installed only on request (/lsp), never silently.
        env.setdefault("OPENCODE_DISABLE_LSP_DOWNLOAD", "1")
        argv = [str(ui_bin)]
        if alias:
            argv += ["-m", f"localcode/{alias}"]
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
