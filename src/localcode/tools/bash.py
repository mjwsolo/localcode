"""bash — run a shell command (with GUI-app detach + backgrounding logic)."""
from __future__ import annotations

import atexit
import os
import re
import shlex
import signal
import subprocess
import tempfile
from pathlib import Path

from .base import ToolContext
from .._subproc_env import clean_env
from ..process_registry import ProcessRecord, record_process


# Every Popen we launch with `start_new_session=True` becomes a session
# leader (pgid == pid). We record the pgid here so the agent loop can
# reap them at session end — without this, every `python3 app.py &`
# the model issues to verify a build leaks a runaway interpreter that
# spins at 100 % CPU until the user notices the battery drain. (It
# happened: 7 zombies, 8 hours each, ~520 min CPU each.)
#
# Process groups stay valid as kill targets even after the leader dies
# (POSIX behaviour, confirmed on Darwin), so the `&` path's grandchild
# is reachable via killpg(wrapper_pgid, …) even after the wrapper
# shell has exited.
_BACKGROUND_PGIDS: set[int] = set()


# AirPlay-collision detection helpers. macOS Monterey+ ships AirPlay
# Receiver bound to port 5000 and AirTunes on 7000; both respond to
# bare HTTP probes in ways that look "alive" to a model running
# `curl -s localhost:5000`, leading the model to conclude its app
# is reachable on those ports when actually AirPlay is replying.
_AIRPLAY_PORTS = (5000, 7000)
_LOCALHOST_PORT_RE = re.compile(
    r"localhost:(\d+)|127\.0\.0\.1:(\d+)|0\.0\.0\.0:(\d+)"
)


def _looks_like_airplay_curl(cmd: str, exit_code: int) -> bool:
    """Heuristic: command ran something against localhost:5000 or :7000.

    Triggers only when the cmd mentions one of those ports; we don't
    want to do a probe for every bash invocation."""
    if "localhost" not in cmd and "127.0.0.1" not in cmd and "0.0.0.0" not in cmd:
        return False
    return any(f":{p}" in cmd for p in _AIRPLAY_PORTS)


def _airplay_check(cmd: str) -> str:
    """If the targeted port has AirPlay's signature `Server` header,
    return a warning the bash result will be prefixed with. Otherwise
    return empty string (no warning surfaced)."""
    m = _LOCALHOST_PORT_RE.search(cmd)
    if not m:
        return ""
    port_str = next((g for g in m.groups() if g), None)
    if not port_str:
        return ""
    try:
        port = int(port_str)
    except ValueError:
        return ""
    if port not in _AIRPLAY_PORTS:
        return ""
    # Probe `Server:` header. AirPlay returns `AirTunes/...` or
    # `AirReceiver`; Flask/Werkzeug/uvicorn return their own. Use a
    # short timeout so we don't make the model wait for slow probes.
    try:
        r = subprocess.run(
            ["curl", "-sI", "-m", "2", f"http://localhost:{port}/"],
            capture_output=True, text=True, timeout=3,
        )
        head = r.stdout.lower()
    except Exception:
        return ""
    is_airplay = (
        "airtunes" in head
        or "airreceiver" in head
        or "airplay" in head
    )
    if not is_airplay:
        return ""
    return (
        f"⚠ AIRPLAY COLLISION DETECTED on port {port}. The response "
        f"you're seeing is from macOS AirPlay Receiver, NOT your app. "
        f"AirPlay returns 200/403 with an empty body, which makes "
        f"`curl | head` look successful but `open http://localhost:"
        f"{port}` shows '403 Access denied'. Your app is either not "
        f"running OR running on a DIFFERENT port. To fix: pick a port "
        f"≥ 5001 (e.g. 5001, 8000, 8080), HARDCODE it in your app's "
        f"`app.run(port=...)` call (NOT a random/dynamic port), then "
        f"restart and `curl localhost:<NEW_PORT>` to verify."
    )


# Generic port-in-use detector. Distinct from the AirPlay one above —
# AirPlay catches "your :5000 curl LOOKS like it's working but isn't",
# this catches "your server failed to bind because something else owns
# the port." Without this, the model sees "Address already in use"
# and tries to start its own server on the same port again. Observed
# 2026-04-26: 40 retries against the same port 63491 in one session.
# Two-stage detection: a "is this a port-bind failure?" signature
# (cheap, no captures), then a port-extraction pass over likely
# locations. Splitting these avoids regex contortions to skip past
# IP fragments like '0.0.0.0' before reaching the real port number.
_PORT_BIND_SIGNATURES = re.compile(
    r"address already in use|EADDRINUSE|\[Errno 48\]|bind:\s*Address already in use",
    re.IGNORECASE,
)
# Port extractors, in order of specificity. The first one that
# matches wins. Each returns a captured port number.
_PORT_EXTRACTORS = (
    # Flask/Werkzeug: "Port 5000 is in use"
    re.compile(r"Port\s+(\d{2,5})\s+is in use", re.IGNORECASE),
    # Node / libuv: "EADDRINUSE :::3000" or "EADDRINUSE 127.0.0.1:3000"
    re.compile(r"EADDRINUSE[^\d]*?:(\d{2,5})\b", re.IGNORECASE),
    # Tuple-style: ('0.0.0.0', 8000) or ('127.0.0.1', 8080)
    re.compile(r"\(\s*['\"][\d.]+['\"]\s*,\s*(\d{2,5})\s*\)"),
    # localhost / 127.0.0.1 / 0.0.0.0 with port
    re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0)[:\s]+(\d{2,5})\b"),
    # Bare "port N" mention
    re.compile(r"\bport\s+(\d{2,5})\b", re.IGNORECASE),
)

_TREE_COMMAND_HINTS = (
    "ls -R",
    "find ",
    "tree",
    "git ls-files",
    "fd ",
)

_STARTUP_COMMAND_HINTS = (
    "python3 main.py",
    "python main.py",
    "uvicorn",
    "flask run",
    "npm run dev",
    "npm start",
    "npm run start",
    "next dev",
    "vite",
    "streamlit run",
)

_SHELL_FILE_READ_COMMANDS = (
    "cat",
    "head",
    "sed",
    "nl",
)

_PACKAGE_INSTALL_RE = re.compile(
    r"(^|\s)(uv\s+pip\s+install|pip3?\s+install|python3?\s+-m\s+pip\s+install|npm\s+install|pnpm\s+install|yarn\s+install|bun\s+install|bundle\s+install|cargo\s+install)\b",
    re.IGNORECASE,
)

_APP_SOURCE_SUFFIXES = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".vue", ".svelte",
    ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php", ".dart",
)


def _port_in_use_check(output: str) -> str:
    """If `output` shows a port-bind failure, return a strong nudge.
    Otherwise empty string. Probes `lsof -i :<port>` so the model knows
    what's holding the port (often the user's own previous run that
    didn't shut down cleanly)."""
    if not _PORT_BIND_SIGNATURES.search(output):
        return ""
    port = None
    for rx in _PORT_EXTRACTORS:
        m = rx.search(output)
        if m:
            try:
                cand = int(m.group(1))
            except (ValueError, IndexError):
                continue
            # Skip implausible values (well-known privileged ports below
            # 80 are rarely bound by user apps and often false positives
            # from version strings like "Errno 48").
            if 80 <= cand <= 65535:
                port = cand
                break
    if port is None:
        return ""
    # Identify the squatter so the diagnostic is actionable.
    holder = ""
    try:
        r = subprocess.run(
            ["lsof", "-iTCP", f":{port}", "-sTCP:LISTEN", "-Pn"],
            capture_output=True, text=True, timeout=2,
        )
        # lsof prints a header + one row per holder. Keep it small so
        # the nudge stays focused.
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if len(lines) >= 2:
            holder = "\n".join(lines[:3])
    except Exception:
        pass
    holder_block = (
        f"\n\nWho's holding port {port}:\n{holder}\n"
        if holder else ""
    )
    suggested = port + 1 if port < 65535 else 8080
    return (
        f"⚠ PORT {port} IS OCCUPIED. Your server failed to start because "
        f"something else is already listening on that port. "
        f"DO NOT retry the same port — that will fail again."
        f"{holder_block}\n"
        f"To fix, pick ONE of:\n"
        f"  (a) Hardcode a different port in your code (e.g. {suggested}, "
        f"8000, 8080) and re-run.\n"
        f"  (b) Kill the existing holder shown above with "
        f"`kill <PID>`, then retry on port {port}.\n"
        f"Tell the user which port you ended up on."
    )


def _summarize_large_tree_output(cmd: str, output: str) -> str:
    """Compress giant directory listings before they hit history.

    Recursive listings and `find` output are useful, but raw trees can
    explode the prompt and make the next round slow enough that the
    model starts thrashing. Keep a compact head/tail summary instead.
    """
    if not output or len(output) < 20_000:
        return output
    lowered = cmd.lower()
    if not any(hint in lowered for hint in _TREE_COMMAND_HINTS):
        return output

    lines = [ln for ln in output.splitlines() if ln.strip()]
    if len(lines) <= 160:
        return output

    head = lines[:80]
    tail = lines[-24:]
    omitted = max(0, len(lines) - len(head) - len(tail))
    return "\n".join([
        f"[truncated directory listing: {len(lines)} lines, {omitted} omitted]",
        *head,
        "...",
        *tail,
    ])


def _shell_words(value: str) -> list[str]:
    try:
        return shlex.split(value)
    except ValueError:
        return []


def _extract_leading_cd(cmd: str) -> tuple[str, str]:
    """Return (repo-relative cwd prefix, command body) for simple `cd x && y`.

    This intentionally handles only the common model pattern. If a shell
    command is complex enough that we cannot parse it confidently, bash
    remains available instead of over-blocking legitimate commands.
    """
    parts = re.split(r"\s+&&\s+", cmd.strip(), maxsplit=1)
    if len(parts) != 2:
        return "", cmd.strip()
    words = _shell_words(parts[0])
    if len(words) == 2 and words[0] == "cd":
        return words[1].strip(), parts[1].strip()
    return "", cmd.strip()


def _safe_read_file_path(repo: str, cd_prefix: str, raw_path: str) -> str:
    raw_path = raw_path.strip()
    if not raw_path or raw_path.startswith(("http://", "https://", "$", "<")):
        return ""
    if any(ch in raw_path for ch in "*?[]{}"):
        return ""
    if raw_path in {"-", "/dev/null"}:
        return ""

    if os.path.isabs(raw_path):
        try:
            rel = os.path.relpath(raw_path, repo)
        except ValueError:
            return ""
        if rel.startswith(".."):
            return ""
        return rel

    if cd_prefix and cd_prefix not in {".", "./"}:
        return os.path.normpath(os.path.join(cd_prefix, raw_path))
    return os.path.normpath(raw_path)


def _file_read_rejection(path: str, offset: int | None = None, limit: int | None = None) -> str:
    args = f"path={path!r}"
    if offset is not None:
        args += f", offset={offset}"
    if limit is not None:
        args += f", limit={limit}"
    return (
        "REJECTED: use read_file instead of bash to inspect file contents.\n"
        f"Suggested tool call: read_file({args}).\n"
        "Bash is for running commands; file reads must use read_file so "
        "LocalCode can cap output, preserve edit context, and avoid prompt bloat."
    )


def _file_write_rejection(path: str) -> str:
    return (
        "REJECTED: use write_file/edit_file/multi_edit instead of bash shell "
        "redirection to write project files.\n"
        f"Target path: {path!r}.\n"
        "Bash is for running commands; file writes must use structured file "
        "tools so LocalCode can track diffs, reject destructive rewrites, "
        "preserve edit context, and verify outcomes."
    )


def _redirect_shell_file_write(cmd: str, repo: str) -> str:
    """Block shell-as-file-writer anti-patterns inside the project.

    This does not block programs that generate files themselves
    (`python script.py`, `npm run build`, etc.). It blocks shell
    redirection/heredoc writes such as `cat > file <<EOF`, `tee file`,
    and `echo ... > file`, which bypass structured edit tracking.
    """
    cd_prefix, body = _extract_leading_cd(cmd)
    stripped = body.strip()
    if not stripped:
        return ""

    patterns = [
        r"^(?:cat|printf|echo)\b.*?(?:>>|>)\s*([^\s;&|]+)",
        r"^tee(?:\s+-a)?\s+([^\s;&|]+)",
    ]
    for pattern in patterns:
        m = re.match(pattern, stripped, re.DOTALL)
        if not m:
            continue
        words = _shell_words(m.group(1))
        raw_path = words[0] if words else ""
        path = _safe_read_file_path(repo, cd_prefix, raw_path)
        if path:
            return _file_write_rejection(path)
    # Generic `command > file` redirection (e.g. `python -c "..." > data.json`,
    # `curl URL > data.json`, `python generate.py > out.json`) is allowed.
    # The earlier catch-all blocked these and forced the model into a
    # two-step "compute → stringify → write_file" loop on what should be
    # one bash call — wasting a turn on every data-import / scratch save.
    # The real antipattern (model TYPING file content into the shell) is
    # already covered by patterns 1 and 2 above (`cat > file`, `echo > file`,
    # `printf > file`, `tee file`).
    return ""


def _redirect_shell_file_read(cmd: str, repo: str) -> str:
    """Block shell-as-file-reader anti-patterns.

    Real regression: edit tasks repeatedly ran `cat -n file | sed -n ...`
    instead of `read_file`, inflating history and making edits slower and
    less accurate. This is generic across languages: source/text file
    inspection belongs to the structured file reader, not bash.
    """
    cd_prefix, body = _extract_leading_cd(cmd)
    lowered = body.strip().lower()
    if not lowered.startswith(_SHELL_FILE_READ_COMMANDS):
        return ""

    # cat -n path | sed -n '10,40p'  OR  nl -ba path | sed -n '10,40p'
    m = re.match(
        r"^(?:cat(?:\s+-n)?|nl\s+-ba)\s+(.+?)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s*$",
        body,
    )
    if m:
        words = _shell_words(m.group(1))
        path = _safe_read_file_path(repo, cd_prefix, words[0] if words else "")
        if path:
            start = max(1, int(m.group(2)))
            end = max(start, int(m.group(3)))
            return _file_read_rejection(path, offset=start - 1, limit=end - start + 1)

    # sed -n '10,40p' path
    m = re.match(r"^sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+(.+?)\s*$", body)
    if m:
        words = _shell_words(m.group(3))
        path = _safe_read_file_path(repo, cd_prefix, words[0] if words else "")
        if path:
            start = max(1, int(m.group(1)))
            end = max(start, int(m.group(2)))
            return _file_read_rejection(path, offset=start - 1, limit=end - start + 1)

    # head -n 80 path
    m = re.match(r"^head\s+-n\s+(\d+)\s+(.+?)\s*$", body)
    if m:
        words = _shell_words(m.group(2))
        path = _safe_read_file_path(repo, cd_prefix, words[0] if words else "")
        if path:
            return _file_read_rejection(path, offset=0, limit=max(1, int(m.group(1))))

    # cat path / cat -n path. Keep this strict so `cat <<EOF` etc. still runs.
    m = re.match(r"^cat(?:\s+-n)?\s+(.+?)\s*$", body)
    if m:
        words = _shell_words(m.group(1))
        if len(words) == 1:
            path = _safe_read_file_path(repo, cd_prefix, words[0])
            if path:
                return _file_read_rejection(path)

    return ""


def _path_within(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _reject_noncanonical_creation_bash(cmd: str, repo: str, current_task: object | None) -> str:
    return ""


def _looks_like_startup_command(cmd: str) -> bool:
    lowered = cmd.lower()
    return any(hint in lowered for hint in _STARTUP_COMMAND_HINTS)


def _has_app_source_files(root: Path) -> bool:
    try:
        if not root.exists() or not root.is_dir():
            return False
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_parts = set(path.relative_to(root).parts)
            if rel_parts & {"node_modules", ".venv", "venv", "__pycache__", ".git", "dist", "build"}:
                continue
            if path.name in {"package.json", "requirements.txt", "pyproject.toml", "uv.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}:
                continue
            if path.suffix.lower() in _APP_SOURCE_SUFFIXES:
                return True
    except Exception:
        return False
    return False


def _reject_premature_dependency_install_for_creation(cmd: str, repo: Path, current_task: object | None) -> str:
    if getattr(current_task, "task_kind", "") != "new_app":
        return ""
    if not _PACKAGE_INSTALL_RE.search(cmd):
        return ""

    # Where to look for source files. The task_slug is what the controller
    # inferred from the user's request, but the model may pick its own
    # directory name.
    # If the bash command starts with `cd <path> && …`, trust that path —
    # it's where the model is actually operating. Fall back to the slug
    # only when no `cd` prefix is present.
    candidate_roots: list[Path] = []
    cd_match = re.match(r"\s*cd\s+(\S+)\s*&&", cmd)
    if cd_match:
        cd_target = cd_match.group(1).strip().strip('"').strip("'")
        if cd_target:
            try:
                cd_path = Path(cd_target)
                if not cd_path.is_absolute():
                    cd_path = repo / cd_path
                candidate_roots.append(cd_path)
            except Exception:
                pass
    slug = str(getattr(current_task, "task_slug", "") or "").strip()
    if slug:
        candidate_roots.append(repo / slug)
    if not candidate_roots:
        return ""

    for root in candidate_roots:
        if _has_app_source_files(root):
            return ""
    return (
        "REJECTED: defer dependency installation until after the new project has "
        "source files. Write the manifest and actual source code first; then run "
        "the install/build/run verification command."
    )


def _command_after_cd_prefix(cmd: str) -> str:
    """Return the command segment that would actually execute after
    common `cd dir && ...` setup. This keeps server detection from
    treating probes like `ps aux | grep app.py` as app launches just
    because they mention a server filename.
    """
    segment = cmd.strip()
    parts = re.split(r"\s+&&\s+", segment)
    while len(parts) > 1 and parts[0].strip().lower().startswith("cd "):
        parts.pop(0)
    return parts[0].strip() if parts else segment


def _looks_like_detached_server_command(cmd: str) -> bool:
    command = _command_after_cd_prefix(cmd)
    lowered = command.lower().strip()
    if not lowered:
        return False

    # Never detach probes/inspection/control commands just because their
    # arguments mention app.py/server.py/npm/etc. Real failure: `ps aux |
    # grep app.py` was backgrounded and returned "The app is running".
    if re.match(
        r"^(?:ps\b|grep\b|curl\b|cat\b|lsof\b|kill\b|pkill\b|head\b|tail\b|"
        r"ls\b|find\b|sed\b|awk\b|rg\b|which\b|whereis\b)",
        lowered,
    ):
        return False
    if " | " in lowered and re.match(r"^(?:ps|grep|curl|cat|lsof)\b", lowered):
        return False
    if re.search(r"\bpython3?\s+-c\b", lowered):
        return False

    return any(pattern.lower() in lowered for pattern in _GUI_PATTERNS)


def _port_is_listening(port: int) -> bool:
    if port <= 0:
        return False
    try:
        r = subprocess.run(
            ["lsof", "-iTCP", f":{port}", "-sTCP:LISTEN", "-Pn"],
            capture_output=True, text=True, timeout=2,
        )
    except Exception:
        return False
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    return len(lines) >= 2


def _extract_command_port(cmd: str, output: str = "") -> int:
    text = f"{cmd}\n{output}"
    for match in re.finditer(r"(?:localhost:|127\.0\.0\.1:|--port\s+|-p\s+)(\d{2,5})", text):
        try:
            port = int(match.group(1))
        except Exception:
            continue
        if 1 <= port <= 65535:
            return port
    return 0


def _normalize_task_port(cmd: str, active_port: int) -> str:
    """Keep build-app launch/verify commands pinned to one port.

    The loop already tries to remember the discovered port, but the
    model kept re-probing 3000/3001 anyway. Rewrite obvious launch and
    localhost probe variants so follow-up commands stay aligned with the
    task's chosen port.
    """
    if active_port <= 0:
        return cmd

    port = str(active_port)
    if re.search(r"\bnpm\s+(?:run\s+)?(?:dev|start)\b", cmd) and not re.search(r"(?:--\s*-p\s+\d+|-p\s+\d+|--port\s+\d+)", cmd):
        cmd = re.sub(r"\bnpm\s+run\s+dev\b", f"npm run dev -- -p {port}", cmd, count=1)
        cmd = re.sub(r"\bnpm\s+start\b", f"npm start -- -p {port}", cmd, count=1)
        cmd = re.sub(r"\bnpm\s+run\s+start\b", f"npm run start -- -p {port}", cmd, count=1)

    if re.search(r"\buvicorn\b", cmd) and not re.search(r"--port\s+\d+", cmd):
        cmd = re.sub(r"\buvicorn\b", f"uvicorn --port {port}", cmd, count=1)

    # Rewrite common localhost probes / port killers that kept drifting
    # back to 3000/3001 even after the task had chosen a port.
    cmd = re.sub(r"(localhost:|127\.0\.0\.1:)(3000|3001)\b", rf"\g<1>{port}", cmd)
    cmd = re.sub(r"(:)(3000|3001)\b", rf":{port}", cmd)
    return cmd


def _track_background(pid: int) -> None:
    if pid:
        _BACKGROUND_PGIDS.add(pid)


def reap_background_processes() -> int:
    """SIGTERM every background process the bash tool launched in this
    session. Idempotent — safe to call from multiple cleanup paths
    (atexit, SIGINT handler, TUI quit). Returns the number of pgids
    we attempted to kill (not all may have been alive)."""
    pgids = list(_BACKGROUND_PGIDS)
    _BACKGROUND_PGIDS.clear()
    killed = 0
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        except Exception:
            pass
    return killed


atexit.register(reap_background_processes)

SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command. Use for: running code, tests, git, "
            "installing packages, launching apps. "
            "Commands already run with the repo root as the current working "
            "directory. Prefer relative paths and `cd subdir`; do NOT invent "
            "or retry guessed `/Users/...` absolute paths unless the user "
            "explicitly provided that path. "
            # Placement rationale: tool-call-time guidance belongs in
            # the tool description, not buried as a prompt rule. The
            # model reads this description every time it decides to
            # use bash, so preferences like "use uv, not pip" land at
            # exactly the right moment.
            "PYTHON INSTALLS: prefer `uv pip install <pkg>` or `uv add "
            "<pkg>` over plain `pip install` — uv is 10-100× faster "
            "and more reliable. `uv venv` over `python -m venv`. Fall "
            "back to pip only if `uv` isn't on PATH. "
            "On macOS specifically: "
            "use `open <url>` to launch a URL in the user's default "
            "browser (DO NOT curl the URL and try to 'show' it — curl "
            "is for probing, `open` is for user-visible launching); use "
            "`open <path>` to reveal a file in Finder. "
            "PORTS — NEVER hardcode a port number; discover a free one "
            "before launching. Apple's AirPlay Receiver squats 5000 and "
            "AirTunes squats 7000 on macOS, so Flask's default `app.run()` "
            "silently 403s through AirPlay. Get a guaranteed-free port "
            "with one bash call: `python3 -c \"import socket; s=socket."
            "socket(); s.bind(('',0)); print(s.getsockname()[1]); "
            "s.close()\"`. Capture that integer, pass it explicitly to "
            "your server (`app.run(port=PORT)`, `uvicorn ... --port PORT`, "
            "etc.), then use the SAME port in the verify-curl and the "
            "final `open http://localhost:PORT` step. "
            "To run a long-lived server without blocking, use "
            "`nohup <cmd> > /tmp/out.log 2>&1 &` and then verify with "
            "`curl -s <url>` — do not restart the server if `curl` "
            "already succeeded once; restarting is how you orphan PIDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}


_GUI_PATTERNS = [
    "pygame", "tkinter", "kivy", "PyQt", "PySide", "wx.", "pong", "game",
    # Server runners — explicit framework CLIs
    "flask run", "uvicorn", "streamlit", "gradio", "serve",
    # Common Python server entry-point filenames. `python3 app.py` is
    # the canonical Flask/FastAPI/Django invocation that doesn't use
    # the framework's own CLI. Without these patterns the command runs
    # in the foreground, 120s timeout fires, the agent is told "Tell
    # the user to run it themselves" — defeating the whole point of
    # the agent. Real failure 2026-04-26: model ran
    # `cd generated-app && python3 app.py`, timed out at 120s.
    "app.py", "server.py", "main.py", "wsgi.py", "asgi.py",
    "manage.py runserver",
    # Node.js servers — `node server.js` / `node index.js` / `node app.js`
    # are the canonical patterns. Stricter than just "node" because
    # `node` is also used for one-off scripts that should NOT background.
    "node server.js", "node app.js", "node index.js",
    # JS package-manager dev/start commands almost always launch a
    # long-lived dev server (vite, next, webpack-dev-server, etc.)
    "npm start", "npm run dev", "npm run start",
    "yarn start", "yarn dev",
    "pnpm start", "pnpm dev", "pnpm run dev",
    "bun run dev", "bun dev",
    # Other framework CLIs
    "rails server", "rails s ",
    "hugo serve", "jekyll serve",
    "php -S",
]

# Detect explicit shell backgrounding: lone `&` (not `&&`)
_BG_RE = re.compile(r"(?<!&)&(?!&)")


def _normalize_repo_root_variants(cmd: str, repo: str) -> str:
    """Rewrite obvious hallucinated absolute paths back to the real repo root.

    Weak local models often preserve the stable tail of the repo path
    while corrupting the username segment under `/Users`. Commands already
    run with `cwd=repo`, so absolute paths are unnecessary; when they do
    appear and clearly target this repo, normalize them to the actual root
    instead of letting the command fail for a fake home-directory variant.
    Then a general pass repairs ANY `/Users/<garbled>/` (not just repo paths)
    back to the real home — the model mangles the username via penalty-induced
    token corruption (`marcsolomon` → `marcolon`/`marcslomon`), a DIFFERENT
    misspelling each time, so it can't be dedup'd; we fix it deterministically.
    """
    if not repo.startswith("/Users/"):
        return _repair_home_username(cmd)
    parts = repo.strip("/").split("/")
    if len(parts) < 3:
        return _repair_home_username(cmd)
    stable_tail = "/".join(parts[2:])
    pattern = re.compile(
        rf"/Users/[^/\s'\"]+/{re.escape(stable_tail)}(?=(?:/|\s|['\"]|$))"
    )
    return _repair_home_username(pattern.sub(repo, cmd))


def _repair_home_username(cmd: str) -> str:
    """Fix `/Users/<wrong>/…` → `/Users/<real>/…` using the actual home dir.

    localcode knows the real username; any `/Users/<name>/` where <name> is not
    the real one AND not another existing home directory is a corruption — swap
    in the real username so the command targets the file the model meant.
    """
    import os as _os
    home = _os.path.expanduser("~")
    if not home.startswith("/Users/"):
        return cmd
    real_user = home.split("/")[2]
    if not real_user:
        return cmd

    def _sub(m):
        name = m.group(1)
        if name == real_user:
            return m.group(0)
        # Leave real, existing OTHER home dirs alone (multi-user machines).
        if _os.path.isdir(f"/Users/{name}"):
            return m.group(0)
        return f"/Users/{real_user}/"

    return re.sub(r"/Users/([^/\s'\"]+)/", _sub, cmd)


def _quoting_error_hint(output: str) -> str:
    """Turn a raw shell quoting failure into actionable guidance.

    A path with spaces or parentheses (e.g. `Qwen 3.6 35B-A3B (Q8)`) that
    isn't quoted breaks bash with `syntax error near unexpected token` — an
    error a model can't act on, so it retries the same broken shape in a loop
    (observed exactly this). Tell it precisely how to fix it.
    """
    low = output.lower()
    if "syntax error near unexpected token" not in low and "unexpected eof" not in low:
        return ""
    return (
        "HINT: a path or argument with spaces or parentheses wasn't quoted, so "
        "the shell mis-parsed it. Wrap paths in SINGLE quotes — e.g. "
        "ls -la '/Users/you/My Dir (v2)'. Simpler and safer: use list_files or "
        "read_file with the path as an argument — they take the raw path and "
        "never need shell escaping."
    )


def _redirect_shell_dir_listing(cmd: str) -> str:
    """Route `ls <path>` to list_files, which can't break on spaces/parens.

    Mirrors the `cat → read_file` guard. Only fires for a plain `ls` with a
    PATH argument (no pipes/redirs/chains) — bare `ls`/`ls -la` in the cwd
    can't hit the quoting failure, so it's left to run.
    """
    _cd, body = _extract_leading_cd(cmd)
    b = body.strip()
    if any(sep in b for sep in ("|", ">", "<", "&&", "||", ";", "$(", "`")):
        return ""
    # Path arg must NOT start with '-' (else `ls -la` reads its own flags as a path).
    m = re.match(r"^ls((?:\s+-[a-zA-Z]+)*)\s+([^-\s].*)$", b)
    if m is None:
        return ""  # bare `ls`/`ls -la` (no path) — harmless, let it run
    path = m.group(2).strip().strip('"').strip("'")
    return (
        f"REJECTED: use list_files(path='{path}') to inspect a directory, not "
        "bash `ls <path>`. list_files takes the path as an argument, so it never "
        "breaks on spaces or parentheses in a folder name — which is what fails "
        "with bash here."
    )


def execute(ctx: ToolContext, args: dict) -> str:
    cmd = _normalize_repo_root_variants(args["command"], str(ctx.repo))
    # Block ANY use of process-attaching debuggers — lldb / dtrace /
    # spindump / sample. The agent kept invoking these to debug perceived
    # hangs, which:
    #   - Triggers macOS Touch ID for "Developer Tools Access"
    #   - SIGSTOPs the parent process (freezes the LocalCode TUI)
    #   - On detach, often leaves the parent unrecoverable; user sees
    #     "zsh: abort" when localcode dies
    # No legitimate localcode workflow needs these — block unconditionally.
    # Word-boundary regex so we don't false-positive on filenames like
    # "sample.txt" or "spindump.log".
    import re as _re
    # ONLY match when one of the debugger names is the COMMAND being
    # executed — the first token at start-of-string OR right after a
    # pipe / && / ; / `bash -c '` / `sh -c "` etc. Previous version
    # false-positived on the word "sample" appearing inside a quoted
    # Python string (e.g. `python3 -c "...sample_rate..."`). The new
    # pattern requires the word to be followed by either whitespace
    # plus a non-alphanumeric (an argument start) or end-of-segment
    # — which `sample_rate` and `samples/` and `sample.txt` all
    # naturally avoid.
    _DEBUGGER_RE = _re.compile(
        r"""(?:^|(?<=[\s;&|`])|(?<=bash\s-c\s')|(?<=sh\s-c\s')|(?<=bash\s-c\s")|(?<=sh\s-c\s"))"""
        r"""(?:sudo\s+)?(?:/[^\s'"]*/)?"""
        r"""(?P<dbg>lldb|dtrace|spindump|sample)"""
        r"""(?=\s+-|\s+[0-9]|\s*$|\s*[;&|`])""",
    )
    if _DEBUGGER_RE.search(cmd):
        return (
            "REJECTED: process-attaching debuggers (lldb / dtrace / spindump / "
            "sample) are blocked. They trigger macOS Touch ID prompts, SIGSTOP "
            "the LocalCode TUI, and leave it unrecoverable when they detach. "
            "Use `~/.local/share/localcode/server.log`, "
            "`~/.localcode/last_error.log`, or the `/status` slash command "
            "to diagnose instead."
        )
    app_session = getattr(ctx.app, "session", None)
    current_task = getattr(app_session, "current_task", None)
    current_task_port = int(getattr(current_task, "active_port", 0) or 0) if current_task is not None else 0
    if current_task is not None and getattr(current_task, "goal_type", "") == "build_app":
        cmd = _normalize_task_port(cmd, current_task_port)
        if current_task_port and _port_is_listening(current_task_port):
            lowered = cmd.lower()
            if "kill -9" in lowered or "kill -term" in lowered or "kill " in lowered and _looks_like_startup_command(cmd):
                return (
                    f"REJECTED: build_app already has a live server on port {current_task_port}. "
                    f"Do not kill/restart it. Reuse the existing server and continue editing/verification "
                    f"against that port."
                )
            if _looks_like_startup_command(cmd):
                return (
                    f"REJECTED: build_app already has a live server on port {current_task_port}. "
                    f"Do not start another copy. Reuse the existing server and verify it instead."
                )
    repo = ctx.repo
    noncanonical_bash = _reject_noncanonical_creation_bash(cmd, str(repo), current_task)
    if noncanonical_bash:
        return noncanonical_bash
    premature_install = _reject_premature_dependency_install_for_creation(cmd, repo, current_task)
    if premature_install:
        return premature_install
    file_read_redirect = _redirect_shell_file_read(cmd, str(repo))
    if file_read_redirect:
        return file_read_redirect
    dir_listing_redirect = _redirect_shell_dir_listing(cmd)
    if dir_listing_redirect:
        return dir_listing_redirect
    file_write_redirect = _redirect_shell_file_write(cmd, str(repo))
    if file_write_redirect:
        return file_write_redirect

    # GUI / long-running servers: launch detached, don't block. We pipe
    # output to a log and WAIT briefly for the server to either print its
    # URL or crash — so we can hand the agent the REAL url (dev servers like
    # vite hop 5173→5174→5175 when a port is taken; returning a hardcoded
    # port is why the user hit a 404) and surface a startup crash as an
    # error the agent can fix, instead of a blind "the app is running".
    is_gui = _looks_like_detached_server_command(cmd)
    if is_gui and "&" not in cmd:
        import time as _time
        fd, log_path = tempfile.mkstemp(prefix="lc-server-", suffix=".log")
        os.close(fd)
        wrapped = f"({cmd}) > {log_path} 2>&1"
        try:
            proc = subprocess.Popen(
                wrapped, shell=True, cwd=str(repo),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=clean_env(),
            )
        except Exception as e:
            return f"Error launching: {e}"
        _track_background(proc.pid)

        # Poll the log for a printed URL or an early crash (up to ~18s —
        # vite/next/etc. usually ready in 1-3s; a crash is near-instant).
        _URL_RE = re.compile(
            r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)\S*"
        )
        url, detected_port, log_tail = None, None, ""
        deadline = _time.monotonic() + 18
        crashed = False
        while _time.monotonic() < deadline:
            if proc.poll() is not None:
                crashed = True
                break
            try:
                log_tail = open(log_path, errors="replace").read()
            except Exception:
                log_tail = ""
            m = _URL_RE.search(log_tail)
            if m:
                detected_port = m.group(1)
                url = f"http://localhost:{detected_port}"
                break
            _time.sleep(0.4)

        if crashed:
            try:
                log_tail = open(log_path, errors="replace").read()
            except Exception:
                pass
            tail = "\n".join(log_tail.splitlines()[-25:]).strip()
            return (
                f"Server exited during startup (exit {proc.returncode}) — it is NOT running. "
                f"Fix the error below and retry; do NOT tell the user it's running.\n"
                f"--- {cmd} ---\n{tail or '(no output captured)'}"
            )

        port = detected_port or _extract_command_port(cmd)
        if port:
            try:
                record_process(
                    repo,
                    ProcessRecord(
                        pid=proc.pid, pgid=proc.pid, port=int(port),
                        url=url or f"http://localhost:{port}",
                        cwd=str(repo), kind="bash-background", command=cmd,
                        log_path=log_path, verified=bool(url),
                        started_at=_time.time(),
                    ),
                )
            except Exception:
                pass
        if url:
            return (
                f"Launched in background (PID {proc.pid}). Server is UP at {url}\n"
                f"Verify with `curl -sS -o /dev/null -w '%{{http_code}}' {url}` "
                f"(expect 200), then tell the user to open {url}. "
                f"Logs: {log_path}. Do NOT relaunch — it's already running."
            )
        return (
            f"Launched in background (PID {proc.pid}); still starting — no URL "
            f"printed within 18s. Check progress with `tail -n 40 {log_path}` "
            f"(it may just be slow, or it may not print a URL). Do NOT relaunch blindly."
        )

    # Single timeout for foreground commands. Earlier this short-
    # circuited to 30s for `python`/`python3`/`node` to catch
    # interactive `input()` scripts — but it ALSO killed legitimate
    # data-analysis runs ("compute X across 11K records" runs 100+s
    # cleanly). Real failure observed 2026-04-26: model spent 4+ min
    # on aggregations that should have completed but the 30s cap was
    # killing them mid-flight. 120s matches bash default; genuinely
    # long-running work should be backgrounded with `&` (which goes
    # through a separate code path with no foreground timeout).
    timeout = 120

    # Explicit shell backgrounding (`cmd &`) — capture_output blocks because
    # the bg process keeps stdout open. Redirect to a temp log and detach.
    if _BG_RE.search(cmd):
        fd, log_path = tempfile.mkstemp(prefix="lc-bash-", suffix=".log")
        os.close(fd)
        wrapped = f"({cmd}) > {log_path} 2>&1"
        proc = None
        try:
            proc = subprocess.Popen(
                wrapped, shell=True, cwd=str(repo),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=clean_env(),
            )
            # Track the pgid BEFORE wait() — `cmd &` makes the wrapper
            # shell exit fast, but the grandchild keeps running with
            # the wrapper's old pgid. That pgid is still a valid kill
            # target after the leader dies, so reap_background_processes
            # can clean it up at session end.
            _track_background(proc.pid)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    rc = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = -9
            try:
                with open(log_path) as f:
                    output = f.read().strip()
            except (FileNotFoundError, OSError):
                output = ""
            if rc != 0 and rc is not None:
                output = f"[exit code {rc}]\n{output}"
            port = _extract_command_port(cmd, output)
            if port:
                try:
                    record_process(
                        repo,
                        ProcessRecord(
                            pid=proc.pid,
                            pgid=proc.pid,
                            port=port,
                            url=f"http://localhost:{port}",
                            cwd=str(repo),
                            kind="bash-background",
                            command=cmd,
                            log_path=log_path,
                            verified=not output.startswith("[exit code "),
                            started_at=__import__("time").time(),
                        ),
                    )
                except Exception:
                    pass
            return output or "Command launched (background processes may still be running — output captured to log)"
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass

    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(repo),
            stdin=subprocess.DEVNULL,
            env=clean_env(),
        )
        output = (r.stdout + r.stderr).strip()
        # Prompt-injection guard. A repo file or web response echoed by the
        # command (`cat README`, `curl …`) can carry "ignore all prior
        # instructions" text — bash was the one tool output that reached
        # the model unguarded (read_file/web_fetch already wrap theirs).
        # bash output is also read by internal heuristics and is the
        # overwhelmingly-common case, so we do NOT blanket-wrap: only when
        # a hostile pattern is actually detected do we fence the output +
        # prepend the warning. Clean output passes through byte-for-byte,
        # preserving the exact format every downstream consumer expects.
        try:
            from ..injection_defense import detect_injection_patterns, wrap_untrusted
            if output and detect_injection_patterns(output):
                output = wrap_untrusted(output, source=f"$ {cmd[:60]}")
        except Exception:
            pass
        if r.returncode != 0:
            output = f"[exit code {r.returncode}]\n{output}"
            # Shell quoting failure (unquoted spaces/parens in a path) → give
            # the model an actionable fix instead of a raw syntax error it loops on.
            _qh = _quoting_error_hint(output)
            if _qh:
                output = _qh + "\n\n" + output
        # AirPlay-collision detector: if the model curls localhost:5000
        # or :7000, those are squatted by macOS AirPlay Receiver /
        # AirTunes — bare `curl -s` sees a 200 with empty body and
        # `head -5` returns nothing useful, so the model concludes
        # "port 5000 works" and `open` lands on AirPlay's 403 page.
        # Repeated incident loop observed 2026-04-26. Detect and
        # explicitly tell the model what's happening so it picks a
        # different port instead of looping.
        try:
            if _looks_like_airplay_curl(cmd, r.returncode):
                airplay_warning = _airplay_check(cmd)
                if airplay_warning:
                    output = airplay_warning + ("\n\n" + output if output else "")
        except Exception:
            pass
        output = _summarize_large_tree_output(cmd, output)
        # Generic port-in-use detector. Runs on every nonzero-exit bash
        # call (cheap regex first, lsof only if a port pattern matched)
        # so we don't probe lsof on commands that don't touch ports.
        try:
            if r.returncode != 0:
                port_warning = _port_in_use_check(output)
                if port_warning:
                    output = port_warning + "\n\n" + output
        except Exception:
            pass
        return output or "all good!"
    except subprocess.TimeoutExpired:
        return (
            f"Command timed out ({timeout}s). This might be a long-running or "
            f"interactive process. Tell the user to run it themselves: {cmd}"
        )
