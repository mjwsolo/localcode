"""Generic app launch contract.

This module is deliberately not a giant stack whitelist. It uses a layered
contract:

1. Detect common project manifests and scripts.
2. Pick a safe free port and a deterministic launch command.
3. Start once, verify once, persist process metadata.
4. Return a compact result the agent can trust.

Unknown stacks fall back to the model instead of guessing in a loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from ._subproc_env import clean_env
from .process_registry import ProcessRecord, latest_live_record, record_process, stop_record


@dataclass(frozen=True)
class LaunchCandidate:
    root: Path
    kind: str
    command: str
    url_path: str = "/"
    build_command: str = ""
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class LaunchResult:
    ok: bool
    message: str
    root: str = ""
    kind: str = ""
    command: str = ""
    pid: int = 0
    port: int = 0
    url: str = ""
    log_path: str = ""
    verified: bool = False
    browser_opened: bool = False
    browser_error: str = ""


def launch_project_app(
    repo_root: Path | str,
    *,
    preferred_port: int = 0,
    open_browser: bool = False,
) -> LaunchResult:
    repo = Path(repo_root).resolve()
    live = latest_live_record(repo)
    if live is not None:
        return LaunchResult(
            ok=True,
            message=f"Existing app process is already healthy at {live.url}.",
            root=live.cwd,
            kind=live.kind,
            command=live.command,
            pid=live.pid,
            port=live.port,
            url=live.url,
            log_path=live.log_path,
            verified=True,
            **_browser_open_fields(live.url, open_browser),
        )
    candidate = detect_launch_candidate(repo)
    if candidate is None:
        return LaunchResult(
            ok=False,
            message=(
                "No deterministic app launcher was found. Falling back to the "
                "agent loop; it should inspect the repo and choose an explicit "
                "run command."
            ),
        )

    port = preferred_port if _port_available(preferred_port) else _free_port()
    command = candidate.command.format(port=port)
    env = clean_env()
    env.update(candidate.env or {})
    env.setdefault("PORT", str(port))
    env.setdefault("BROWSER", "none")

    processes_dir = repo / ".localcode" / "processes"
    processes_dir.mkdir(parents=True, exist_ok=True)
    log_path = processes_dir / f"app-{int(time.time())}.log"

    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(candidate.root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    url = f"http://localhost:{port}{candidate.url_path}"
    verified = _wait_for_http(url, timeout_s=12)
    record_process(
        repo,
        ProcessRecord(
            pid=proc.pid,
            pgid=proc.pid,
            port=port,
            url=url,
            cwd=str(candidate.root),
            kind=candidate.kind,
            command=command,
            log_path=str(log_path),
            verified=verified,
            started_at=time.time(),
        ),
    )

    if verified:
        browser_fields = _browser_open_fields(url, open_browser)
        return LaunchResult(
            ok=True,
            message=f"App launched and verified at {url}.",
            root=str(candidate.root),
            kind=candidate.kind,
            command=command,
            pid=proc.pid,
            port=port,
            url=url,
            log_path=str(log_path),
            verified=True,
            **browser_fields,
        )

    return LaunchResult(
        ok=False,
        message=(
            f"Launch command started but did not verify within 12s: {command}. "
            f"Log: {log_path}"
        ),
        root=str(candidate.root),
        kind=candidate.kind,
        command=command,
        pid=proc.pid,
        port=port,
        url=url,
        log_path=str(log_path),
        verified=False,
    )


def stop_project_app(repo_root: Path | str) -> LaunchResult:
    repo = Path(repo_root).resolve()
    live = latest_live_record(repo)
    if live is None:
        return LaunchResult(ok=True, message="No live LocalCode-managed app process found.")
    stopped = stop_record(repo, live)
    return LaunchResult(
        ok=stopped,
        message="Stopped app process." if stopped else "Could not stop app process.",
        root=live.cwd,
        kind=live.kind,
        command=live.command,
        pid=live.pid,
        port=live.port,
        url=live.url,
        log_path=live.log_path,
        verified=False,
    )


def restart_project_app(
    repo_root: Path | str,
    *,
    preferred_port: int = 0,
    open_browser: bool = False,
) -> LaunchResult:
    stopped = stop_project_app(repo_root)
    if not stopped.ok:
        return stopped
    return launch_project_app(
        repo_root,
        preferred_port=preferred_port or stopped.port,
        open_browser=open_browser,
    )


def _browser_open_fields(url: str, open_browser: bool) -> dict[str, object]:
    if not open_browser or not url:
        return {"browser_opened": False, "browser_error": ""}
    ok, error = _open_url(url)
    return {"browser_opened": ok, "browser_error": error}


def _open_url(url: str) -> tuple[bool, str]:
    if not url.startswith(("http://localhost:", "http://127.0.0.1:")):
        return False, "Refused to open non-local URL"
    if os.name == "posix" and sys_platform_darwin():
        cmd = ["open", url]
    elif os.name == "nt":
        cmd = ["cmd", "/c", "start", "", url]
    else:
        cmd = ["xdg-open", url]
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def sys_platform_darwin() -> bool:
    import sys
    return sys.platform == "darwin"


def detect_launch_candidate(repo_root: Path | str) -> LaunchCandidate | None:
    repo = Path(repo_root).resolve()
    roots = _candidate_roots(repo)
    scored: list[tuple[int, LaunchCandidate]] = []
    for root in roots:
        candidate = _detect_node(root) or _detect_python(root) or _detect_static(root)
        if candidate is None:
            continue
        scored.append((_score_root(repo, root), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _candidate_roots(repo: Path) -> list[Path]:
    roots = [repo]
    ignored = {
        "site",
        "docs",
        "tests",
        "benchmarks",
        "eval",
        "src",
        "logo",
        "node_modules",
        ".venv",
    }
    try:
        children = [
            p for p in repo.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in ignored
        ]
    except OSError:
        children = []
    roots.extend(children)
    # Common monorepo/frontend nesting without walking the whole repo.
    for child in list(children):
        for name in ("frontend", "web", "app", "client"):
            nested = child / name
            if nested.is_dir():
                roots.append(nested)
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _score_root(repo: Path, root: Path) -> int:
    score = 0
    if root == repo:
        score += 10
    if root.name.lower() in {"frontend", "web", "app", "client"}:
        score += 20
    try:
        score += int(root.stat().st_mtime) // 1000
    except OSError:
        pass
    return score


def _detect_node(root: Path) -> LaunchCandidate | None:
    package_json = root / "package.json"
    if not package_json.is_file():
        return None
    try:
        data = json.loads(package_json.read_text(errors="replace"))
    except Exception:
        return None
    scripts = data.get("scripts") if isinstance(data, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}
    deps = {}
    if isinstance(data, dict):
        for key in ("dependencies", "devDependencies"):
            if isinstance(data.get(key), dict):
                deps.update(data[key])
    if "vite" in deps or "dev" in scripts and "vite" in str(scripts.get("dev", "")):
        return LaunchCandidate(root=root, kind="node-vite", command="npm run dev -- --host 127.0.0.1 --port {port}")
    if "next" in deps:
        return LaunchCandidate(root=root, kind="node-next", command="npm run dev -- -H 127.0.0.1 -p {port}")
    if "react-scripts" in deps:
        return LaunchCandidate(root=root, kind="node-react-scripts", command="npm start", env={"HOST": "127.0.0.1"})
    if "dev" in scripts:
        return LaunchCandidate(root=root, kind="node-dev", command="npm run dev -- --port {port}")
    if "start" in scripts:
        return LaunchCandidate(root=root, kind="node-start", command="npm start")
    return None


def _detect_python(root: Path) -> LaunchCandidate | None:
    for filename in ("app.py", "main.py", "server.py"):
        path = root / filename
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")[:20_000]
        if "streamlit" in text:
            return LaunchCandidate(root=root, kind="python-streamlit", command=f"streamlit run {filename} --server.port {{port}} --server.headless true")
        if "uvicorn" in text or "FastAPI(" in text:
            module = filename[:-3]
            return LaunchCandidate(root=root, kind="python-asgi", command=f"python3 -m uvicorn {module}:app --host 127.0.0.1 --port {{port}}")
        if "Flask(" in text or "app.run(" in text:
            module = filename[:-3]
            return LaunchCandidate(root=root, kind="python-flask", command=f"python3 -m flask --app {module} run --host 127.0.0.1 --port {{port}}")
    return None


def _detect_static(root: Path) -> LaunchCandidate | None:
    if (root / "index.html").is_file():
        return LaunchCandidate(root=root, kind="static-http", command="python3 -m http.server {port} --bind 127.0.0.1")
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_available(port: int) -> bool:
    if port <= 0 or port > 65535:
        return False
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_http(url: str, *, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if 200 <= int(resp.status) < 500:
                    return True
        except Exception:
            time.sleep(0.35)
    return False
