"""App/build/run task helpers for the agent loop."""
from __future__ import annotations

from pathlib import Path
import re


APP_BUILD_RE = re.compile(
    r"\b(?:build|make|create|start|scaffold|help me build)\b.{0,80}\b"
    r"(?:app|website|site|dashboard|frontend|web app|react app|streamlit|flask app)\b",
    re.IGNORECASE | re.DOTALL,
)
PARTIAL_HANDOFF_RE = re.compile(
    r"(?:\bnext steps\b|\bimplemented features so far\b|\bi have started building\b|"
    r"\bready to proceed\b|\bready to continue\b|\barchitecture overview\b|"
    r"\bwould you like me to\b|\bI'?m ready to proceed\b)",
    re.IGNORECASE,
)
BLOCKING_QUESTION_RE = re.compile(r"^(?:[^?]{0,320}\?)$", re.DOTALL)
PORT_RE = re.compile(r"(?:--port\s+|-p\s+|localhost:|127\.0\.0\.1:)(\d{2,5})")
__all__ = [
    "is_app_build_request",
    "looks_like_partial_handoff",
    "is_focused_blocking_question",
    "extract_port",
    "has_runtime_verification_signal",
    "app_source_line_stats",
    "has_launch_signal",
    "ground_run_or_launch_text",
    "format_run_or_launch_summary",
]


def is_app_build_request(user_text: str) -> bool:
    return bool(APP_BUILD_RE.search(user_text or ""))


def looks_like_partial_handoff(content: str) -> bool:
    return bool(PARTIAL_HANDOFF_RE.search(content or ""))


def is_focused_blocking_question(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if text.count("?") != 1:
        return False
    if not BLOCKING_QUESTION_RE.match(text):
        return False
    lower = text.lower()
    if re.match(r"^(?:hi|hello|hey|yo)[!.\s,]*(?:how can i help|what can i do)", lower):
        return False
    return not any(
        bad in lower for bad in (
            "next steps",
            "ready to proceed",
            "implemented features so far",
            "architecture overview",
        )
    )


def extract_port(text: str) -> int:
    for raw in PORT_RE.findall(text or ""):
        try:
            port = int(raw)
        except Exception:
            continue
        if 1 <= port <= 65535:
            return port
    return 0


def has_runtime_verification_signal(bash_history: list[tuple[str, str]]) -> bool:
    for cmd, result in bash_history:
        cmd_l = (cmd or "").lower()
        result_l = (result or "").lower()
        if result_l.startswith("error:") or result_l.startswith("rejected:"):
            continue
        if "curl " in cmd_l or "http://localhost:" in cmd_l or "http://127.0.0.1:" in cmd_l:
            if any(
                bad in result_l for bad in (
                    "address already in use",
                    "error while attempting to bind",
                    "failed to start",
                    "connection refused",
                    "not found",
                )
            ):
                continue
            return True
        if "open http://localhost:" in cmd_l or "open http://127.0.0.1:" in cmd_l:
            return True
    return False


def app_source_line_stats(repo_root: Path | str, changed_files: list[str]) -> tuple[int, int]:
    repo = Path(repo_root)
    source_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css"}
    source_count = 0
    total_lines = 0
    for rel in changed_files:
        try:
            path = repo / rel
            if path.suffix.lower() not in source_exts or not path.is_file():
                continue
            source_count += 1
            total_lines += len(path.read_text(errors="replace").splitlines())
        except Exception:
            continue
    return source_count, total_lines


def has_launch_signal(bash_history: list[tuple[str, str]]) -> bool:
    for cmd, result in bash_history:
        cmd_l = (cmd or "").lower()
        result_l = (result or "").lower()
        if result_l.startswith("error:") or result_l.startswith("rejected:"):
            continue
        if any(
            token in cmd_l for token in (
                "npm run dev",
                "vite",
                "uvicorn",
                "flask run",
                "python -m http.server",
                "streamlit run",
            )
        ):
            if any(
                bad in result_l for bad in (
                    "exit code 1",
                    "address already in use",
                    "failed to start",
                    "connection refused",
                    "error:",
                )
            ):
                continue
            return True
    return False


def ground_run_or_launch_text(text: str, port: int) -> str:
    if port <= 0 or not text:
        return text
    grounded = text
    grounded = grounded.replace("http://localhost:[FRONTEND_PORT]", f"http://localhost:{port}")
    grounded = grounded.replace("http://127.0.0.1:[FRONTEND_PORT]", f"http://127.0.0.1:{port}")
    grounded = grounded.replace("http://localhost:[PORT]", f"http://localhost:{port}")
    grounded = grounded.replace("http://127.0.0.1:[PORT]", f"http://127.0.0.1:{port}")
    grounded = grounded.replace("localhost:[FRONTEND_PORT]", f"localhost:{port}")
    grounded = grounded.replace("127.0.0.1:[FRONTEND_PORT]", f"127.0.0.1:{port}")
    grounded = grounded.replace("localhost:[PORT]", f"localhost:{port}")
    grounded = grounded.replace("127.0.0.1:[PORT]", f"127.0.0.1:{port}")
    return grounded


def format_run_or_launch_summary(port: int, verified: bool) -> str:
    if port > 0:
        if verified:
            return f"The app is now running and verified.\n\nOpen it at http://localhost:{port}."
        return f"The app is now running.\n\nOpen it at http://localhost:{port}."
    if verified:
        return "The app is now running and verified."
    return "The app is now running."

