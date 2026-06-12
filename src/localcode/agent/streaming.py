"""Streaming helpers for the agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any, TYPE_CHECKING

from .constants import MAX_OUTPUT_TOKENS, MAX_THINKING_CHARS, MAX_THINKING_SECONDS
from ..tools import ALL_SCHEMAS as TOOL_SCHEMAS

if TYPE_CHECKING:
    from ..app import LocalCodeApp
    from ..output import OutputManager


# ── Incremental disk-write helpers ─────────────────────────────────────
#
# When the model is mid-decode on a `write_file` / `append_file` call,
# its tool-call args stream in as a partial JSON blob. We extract the
# `path` field (small, decoded fast) and the `content` field (huge,
# decodes over minutes) and write the file to disk INCREMENTALLY as
# bytes arrive. Result:
#   • The user's editor / file viewer can see the file populating in
#     real time — no waiting 5 min for one big atomic commit.
#   • If localcode crashes / the round errors / the user Ctrl-Cs mid-
#     decode, the partial file is on disk and recoverable. Better than
#     having nothing.
#   • The final atomic write_file.execute() at round-commit becomes
#     idempotent — file already has the same content, so it returns
#     success without re-writing.

_PATH_FIELD_RE = re.compile(r'"path"\s*:\s*"([^"\\]{1,300})"')
_CONTENT_FIELD_RE = re.compile(r'"content"\s*:\s*"')
__all__ = [
    "StreamRoundResult",
    "stream_model_round",
    "finish_thinking_display",
]


def _decode_streaming_content(args: str, start_pos: int) -> tuple[str, int]:
    """Decode JSON-escaped chars from args[start_pos:] until unescaped `"`
    or end-of-string. Restartable across chunks: returns the position
    where decoding paused so the next call resumes there safely (handles
    dangling backslash + incomplete \\uXXXX without losing bytes).

    If `start_pos` is 0, locates the `"content":` field marker first;
    otherwise resumes from inside the string."""
    if start_pos <= 0:
        m = _CONTENT_FIELD_RE.search(args)
        if not m:
            return ("", 0)
        i = m.end()
    else:
        i = start_pos
    n = len(args)
    out: list[str] = []
    while i < n:
        c = args[i]
        if c == '"':
            return ("".join(out), i)
        if c == "\\":
            if i + 1 >= n:
                return ("".join(out), i)
            nxt = args[i + 1]
            if nxt == "n": out.append("\n"); i += 2
            elif nxt == "t": out.append("\t"); i += 2
            elif nxt == "r": out.append("\r"); i += 2
            elif nxt == "b": out.append("\b"); i += 2
            elif nxt == "f": out.append("\f"); i += 2
            elif nxt == '"': out.append('"'); i += 2
            elif nxt == "\\": out.append("\\"); i += 2
            elif nxt == "/": out.append("/"); i += 2
            elif nxt == "u":
                if i + 5 >= n:
                    return ("".join(out), i)
                hex_s = args[i + 2 : i + 6]
                try:
                    out.append(chr(int(hex_s, 16)))
                except ValueError:
                    out.append(args[i : i + 6])
                i += 6
            else:
                out.append(c); i += 1
        else:
            out.append(c); i += 1
    return ("".join(out), i)


def _maybe_incremental_write(
    app: "LocalCodeApp",
    result: "StreamRoundResult",
    *,
    tool_name: str,
    tool_idx: int,
    snippet: str,
) -> None:
    """Write the streaming `content` to disk incrementally for write_file
    / append_file. Tracks per-tool-index state on `result.live_writes`
    so we only write NEW bytes each tick.

    Best-effort: any failure (path not yet decoded, decode error, FS
    error) silently no-ops — the round-end atomic write_file.execute()
    is still authoritative."""
    if tool_name not in {"write_file", "append_file"}:
        return
    if not snippet:
        return
    state = result.live_writes.get(tool_idx)
    if state is None:
        # Need a path before we can write anywhere. Extract it from the
        # leading args; tool schemas put it FIRST so it shows up in the
        # first ~256 chars.
        m = _PATH_FIELD_RE.search(snippet)
        if not m:
            return
        path_str = m.group(1)
        try:
            target = Path(path_str)
            if not target.is_absolute():
                target = Path(app.repo_root) / target
            target.parent.mkdir(parents=True, exist_ok=True)
            # write_file truncates; append_file extends an existing file.
            mode = "w" if tool_name == "write_file" else "a"
            target.open(mode).close()  # truncate / touch
        except Exception:
            return
        state = {"path": str(target), "decoded_pos": 0}
        result.live_writes[tool_idx] = state
    decoded, new_pos = _decode_streaming_content(snippet, state["decoded_pos"])
    if not decoded and new_pos == state["decoded_pos"]:
        return
    if decoded:
        try:
            with open(state["path"], "a", encoding="utf-8") as f:
                f.write(decoded)
        except Exception:
            return
    state["decoded_pos"] = new_pos


@dataclass
class StreamRoundResult:
    content_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    thinking_shown: bool = False
    content_streaming: bool = False
    thinking_abort: bool = False
    finish_reason: str = ""
    raw_tail: str = ""
    content_chars: int = 0
    reasoning_chars: int = 0
    pending_tool_count: int = 0
    ttft_ms: int = 0
    decode_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_estimated: bool = False
    tool_args_limited: bool = False
    limited_tool_name: str = ""
    limited_args_chars: int = 0
    limited_args_snippet: str = ""
    limited_reason: str = ""
    stream_error: str = ""
    live_writes: dict[int, dict] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


def stream_model_round(
    app: "LocalCodeApp",
    out: "OutputManager",
    model_messages: list[dict[str, Any]],
    *,
    round_use_thinking: bool,
    retry_messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None = None,
    recovery_mode: str = "",
    stream_policy: str = "",
) -> StreamRoundResult:
    """Stream one model round and collect text/tool-call diagnostics."""
    from ..runtime import _strip_thinking_tokens

    result = StreamRoundResult()
    stream_start = time.monotonic()
    schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS

    def handle_event(event: dict[str, Any]) -> None:
        nonlocal result
        typ = event.get("type")
        if typ == "stream_done":
            result.finish_reason = event.get("finish_reason", "") or ""
            result.raw_tail = event.get("raw_tail", "") or ""
            result.content_chars = int(event.get("content_chars", 0) or 0)
            result.reasoning_chars = int(event.get("reasoning_chars", 0) or 0)
            result.pending_tool_count = int(event.get("pending_tool_count", 0) or 0)
            result.ttft_ms = int(event.get("ttft_ms", 0) or 0)
            result.decode_ms = int(event.get("decode_ms", 0) or 0)
            pt = int(event.get("prompt_tokens", 0) or 0)
            ct = int(event.get("completion_tokens", 0) or 0)
            tt = int(event.get("total_tokens", 0) or 0)
            result.prompt_tokens = pt
            result.completion_tokens = ct
            result.total_tokens = tt or (pt + ct if (pt or ct) else 0)
            result.usage_estimated = bool(event.get("usage_estimated", False))
            result.tool_args_limited = bool(event.get("tool_args_limited", False))
            result.limited_tool_name = str(event.get("limited_tool_name", "") or "")
            result.limited_args_chars = int(event.get("limited_args_chars", 0) or 0)
            result.limited_args_snippet = str(event.get("limited_args_snippet", "") or "")
            result.limited_reason = str(event.get("limited_reason", "") or "")
            if pt or ct or tt:
                try:
                    out.update_turn_tokens(pt, ct, result.total_tokens)
                except Exception:
                    pass
            return
        if typ == "thinking":
            chunk = _strip_thinking_tokens(event.get("content", ""))
            if not chunk:
                return
            result.thinking_shown = True
            result.thinking_parts.append(chunk)
            out.feed_thinking(chunk)
            if result.content_streaming or result.tool_calls:
                return
            from ..features import Feature, is_enabled
            if is_enabled(Feature.THINKING_CAPS):
                thinking_chars = sum(len(p) for p in result.thinking_parts)
                if (
                    time.monotonic() - stream_start > MAX_THINKING_SECONDS
                    or thinking_chars > MAX_THINKING_CHARS
                ):
                    result.thinking_abort = True
            return
        if typ == "content":
            chunk = _strip_thinking_tokens(event.get("content", ""))
            if not chunk:
                return
            if not result.content_streaming:
                if result.thinking_parts:
                    thinking_text = "".join(result.thinking_parts).strip()
                    if thinking_text:
                        out.thinking_done(thinking_text)
                out.start_streaming()
                result.content_streaming = True
            result.content_parts.append(chunk)
            out.stream(chunk)
            return
        if typ == "tool_preview":
            out.tool_preview(
                event.get("name", ""),
                int(event.get("args_chars", 0) or 0),
                event.get("args_snippet", "") or "",
            )
            _maybe_incremental_write(
                app,
                result,
                tool_name=event.get("name", "") or "",
                tool_idx=int(event.get("index", -1) or -1),
                snippet=event.get("args_snippet", "") or "",
            )
            return
        if typ == "tool_calls":
            result.tool_calls = event.get("tool_calls") or []

    try:
        for event in app.engine.stream_chat_events(
            model_messages,
            tools=schemas,
            think=round_use_thinking,
            num_predict=MAX_OUTPUT_TOKENS,
            recovery_mode=recovery_mode,
            stream_policy=stream_policy,
        ):
            handle_event(event)
            if result.thinking_abort:
                break
        return result
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        has_images = any(
            isinstance(m.get("content"), list) or "images" in m
            for m in retry_messages
        )
        if not has_images:
            result.stream_error = f"{type(exc).__name__}:{exc}"
            raise

        retry_msgs: list[dict[str, Any]] = []
        for m in retry_messages:
            if isinstance(m.get("content"), list):
                text_parts = [
                    p.get("text", "")
                    for p in m["content"]
                    if p.get("type") == "text"
                ]
                retry_msgs.append({"role": m["role"], "content": " ".join(text_parts)})
            elif "images" in m:
                retry_msgs.append({"role": m["role"], "content": m.get("content", "")})
            else:
                retry_msgs.append(m)
        try:
            for event in app.engine.stream_chat_events(
                retry_msgs,
                tools=schemas,
                think=round_use_thinking,
                num_predict=MAX_OUTPUT_TOKENS,
                recovery_mode=recovery_mode,
                stream_policy=stream_policy,
            ):
                handle_event(event)
                if result.thinking_abort:
                    break
            if not result.content_parts and not result.tool_calls:
                out.print_info("Note: image support requires a vision-enabled model.")
            return result
        except Exception as retry_exc:
            result.stream_error = f"{type(retry_exc).__name__}:{retry_exc}"
            raise exc


def finish_thinking_display(result: StreamRoundResult, out: "OutputManager") -> None:
    """Close any non-streamed thinking block through OutputManager."""
    if not result.thinking_parts or result.content_streaming:
        return
    thinking_text = "".join(result.thinking_parts).strip()
    if not thinking_text:
        return
    out.thinking_done(thinking_text)
