"""Structured facts extracted from string tool results.

Tools still return human-readable text for the model and UI, but the
agent loop also needs stable facts for telemetry and lifecycle policy.
Keep extraction generic and conservative: if a value is not explicit in
the result or args, omit it.
"""
from __future__ import annotations

import re
from typing import Any


_URL_RE = re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})(?:/[^\s]*)?")
_EXIT_RE = re.compile(r"^\[exit code (-?\d+)\]", re.MULTILINE)
_PID_RE = re.compile(r"\bPID:\s*(\d+)\b")
_PORT_RE = re.compile(r"(?:localhost:|127\.0\.0\.1:|0\.0\.0\.0:|port\s+)(\d{2,5})", re.IGNORECASE)


def extract_tool_facts(tool_name: str, args: dict[str, Any], result: str) -> dict[str, Any]:
    text = result or ""
    facts: dict[str, Any] = {
        "tool": tool_name,
        "ok": not _looks_error(text),
    }
    if not facts["ok"]:
        facts["error_type"] = _error_type(text)
    if tool_name == "bash":
        command = str(args.get("command", "") or "")
        if command:
            facts["command"] = command
        exit_match = _EXIT_RE.search(text)
        facts["exit_code"] = int(exit_match.group(1)) if exit_match else 0
        if facts["exit_code"] == 0 and _verification_signal(text):
            facts["verification_signal"] = True
        if "launched in background" in text.lower() or "command launched" in text.lower():
            facts["background"] = True
        _add_url_port(facts, f"{command}\n{text}")
    elif tool_name == "launch_app":
        action = str(args.get("action") or "start")
        facts["action"] = action
        facts["verified"] = "App launched and verified" in text
        if facts["verified"]:
            facts["verification_signal"] = True
        if "App process stopped" in text:
            facts["stopped"] = True
        if "Browser opened: yes" in text:
            facts["browser_opened"] = True
        elif "Browser opened: no" in text:
            facts["browser_opened"] = False
        pid_match = _PID_RE.search(text)
        if pid_match:
            facts["pid"] = int(pid_match.group(1))
        _add_url_port(facts, text)
    elif tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}:
        path = args.get("path") or args.get("file_path")
        if isinstance(path, str) and path:
            facts["path"] = path
        facts["changed"] = facts["ok"] and not text.startswith("Error:")
        if "File was reverted" in text or "reverted" in text.lower():
            facts["reverted"] = True
            facts["changed"] = False
    elif tool_name == "read_file":
        path = args.get("path") or args.get("file_path")
        if isinstance(path, str) and path:
            facts["path"] = path
        facts["chars"] = len(text)
        facts["summarized"] = "Large file summarized" in text
    elif tool_name in {"grep", "glob", "list_files", "web_search", "web_fetch", "skill"}:
        path = args.get("path") or args.get("file_path") or args.get("pattern") or args.get("url")
        if isinstance(path, str) and path:
            facts["path"] = path
        facts["chars"] = len(text)
    return {k: v for k, v in facts.items() if v not in ("", None)}


def facts_suffix(facts: dict[str, Any]) -> str:
    public = {
        k: v for k, v in facts.items()
        if k in {"ok", "exit_code", "background", "verified", "verification_signal", "stopped", "browser_opened", "url", "port", "pid", "path", "changed", "summarized", "error_type", "reverted"}
    }
    if not public:
        return ""
    parts = ", ".join(f"{k}={v}" for k, v in public.items())
    return f"\n\n[tool facts: {parts}]"


def _looks_error(text: str) -> bool:
    lowered = text.lower()
    return (
        text.startswith("[exit code ")
        or text.startswith("Error:")
        or text.startswith("REJECTED:")
        or text.startswith("File not found:")
        or "launcher could not verify" in lowered
    )


def _error_type(text: str) -> str:
    lowered = text.lower()
    if text.startswith("REJECTED:"):
        return "rejected"
    if text.startswith("[exit code "):
        if "syntaxerror" in lowered:
            return "syntax_error"
        if "modulenotfounderror" in lowered or "no module named" in lowered:
            return "missing_dependency"
        if "address already in use" in lowered or "eaddrinuse" in lowered:
            return "port_in_use"
        return "nonzero_exit"
    if text.startswith("File not found:"):
        return "not_found"
    if "old_string not found" in lowered:
        return "edit_anchor_not_found"
    if "reverted" in lowered:
        return "reverted"
    if "launcher could not verify" in lowered:
        return "verification_failed"
    if text.startswith("Error:"):
        return "tool_error"
    return "unknown"


def _verification_signal(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "build succeeded",
            "compiled successfully",
            "tests passed",
            "passed",
            "<!doctype html",
            "<html",
            "200 ok",
            "app launched and verified",
        )
    )


def _add_url_port(facts: dict[str, Any], text: str) -> None:
    url_match = _URL_RE.search(text)
    if url_match:
        facts["url"] = url_match.group(0)
        facts["port"] = int(url_match.group(1))
        return
    port_match = _PORT_RE.search(text)
    if port_match:
        facts["port"] = int(port_match.group(1))
