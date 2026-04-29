"""launch_app — deterministic local app launcher."""
from __future__ import annotations

from .base import ToolContext
from ..launcher import launch_project_app, restart_project_app, stop_project_app


SCHEMA = {
    "type": "function",
    "function": {
        "name": "launch_app",
        "description": (
            "Start a local app using LocalCode's deterministic launcher. "
            "Use this before bash when the user says run/start/launch/open "
            "the app. It detects common project manifests, starts one "
            "background process, verifies one localhost URL, records the "
            "PID/port/log, opens the browser when verified, and returns the "
            "exact URL. If it cannot detect the project, fall back to "
            "inspecting files and using bash."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "start, restart, or stop. Default start.",
                },
                "preferred_port": {
                    "type": "integer",
                    "description": "Optional port to reuse for this task if already known.",
                },
                "open_browser": {
                    "type": "boolean",
                    "description": "Open the verified local URL in the user's browser. Default true.",
                },
            },
            "required": [],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    preferred_port = 0
    try:
        preferred_port = int(args.get("preferred_port") or 0)
    except Exception:
        preferred_port = 0
    open_browser = bool(args.get("open_browser", True))
    action = str(args.get("action") or "start").strip().lower()
    if action == "stop":
        result = stop_project_app(ctx.repo)
    elif action == "restart":
        result = restart_project_app(
            ctx.repo,
            preferred_port=preferred_port,
            open_browser=open_browser,
        )
    else:
        result = launch_project_app(
            ctx.repo,
            preferred_port=preferred_port,
            open_browser=open_browser,
        )
    if result.port:
        try:
            if hasattr(ctx.app, "store") and getattr(ctx.app, "session", None) is not None:
                ctx.app.store.update_task(
                    ctx.app.session,
                    active_port=result.port,
                    current_stage="verified" if result.verified else "running",
                )
        except Exception:
            pass
    if result.ok:
        if action == "stop":
            return f"App process stopped.\nPID: {result.pid}\nURL was: {result.url}"
        return (
            f"App launched and verified.\n"
            f"URL: {result.url}\n"
            f"Command: {result.command}\n"
            f"PID: {result.pid}\n"
            f"Log: {result.log_path}\n"
            f"Browser opened: {'yes' if result.browser_opened else 'no'}"
            + (f"\nBrowser error: {result.browser_error}" if result.browser_error else "")
        )
    return (
        f"Launcher could not verify the app.\n"
        f"Reason: {result.message}\n"
        f"Command: {result.command}\n"
        f"URL: {result.url}\n"
        f"Log: {result.log_path}"
    )
