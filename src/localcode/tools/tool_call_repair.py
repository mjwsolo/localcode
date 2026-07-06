"""Recover tool calls a quantized model emitted but the server failed to parse.

Small local models sometimes emit tool calls in a slightly off shape that
llama-server's --jinja parser doesn't recognize, so they come back as plain
assistant CONTENT with zero structured tool_calls: the call never executes (a
wasted round), the raw markup pollutes context, and leaked reasoning makes the
model narrate about "the user". This is the fallback the deleted tool_parsing.py
used to provide — restored here and made comprehensive across ALL model
families, because every family's format leaks the same way with no parser.

Handles (verified Qwen from real logs; others from each family's chat template):
  Qwen  XML : <tool_call><function=NAME><parameter=KEY>\nVALUE\n</parameter>…</function></tool_call>
  Qwen  JSON: <tool_call>{"name": "NAME", "arguments": {…}}</tool_call>
  Gemma     : <|tool_call>call:NAME{…}<tool_call|>
  Cohere    : <|START_ACTION|>[{"tool_name": "NAME", "parameters": {…}}]<|END_ACTION|>

Reasoning strippers cover every family: <think>…</think> (Qwen/DeepSeek),
<unused25>/<|channel>thought (Gemma), <|START_THINKING|>…<|END_THINKING|> (Cohere).
"""
from __future__ import annotations

import json
import re

# ── tool-call shapes ──
_QWEN_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_NAME = re.compile(r"<function\s*=\s*([A-Za-z0-9_]+)\s*>", re.DOTALL)
_PARAM = re.compile(r"<parameter\s*=\s*([A-Za-z0-9_]+)\s*>(.*?)</parameter>", re.DOTALL)
_GEMMA_CALL = re.compile(r"<\|tool_call\>\s*call\s*:\s*(\w+)\s*(\{.*?\})\s*<\s*tool_call\s*\|>", re.DOTALL)
_COHERE_ACTION = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)

# ── reasoning (every family) ──
_THINK_BLOCKS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<unused25>.*?<unused25>", re.DOTALL),
    re.compile(r"<\|START_THINKING\|>.*?<\|END_THINKING\|>", re.DOTALL),
    re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL),
]
# Streaming can drop the OPEN marker, leaving reasoning + a stray CLOSE: drop up
# to and including the first close marker.
_THINK_LEAD = re.compile(r"^.*?(?:</think>|<\|END_THINKING\|>|<unused25>|<channel\|>)", re.DOTALL)
_STRAY = [re.compile(p) for p in (r"</think>", r"<think>", r"<unused25>",
                                  r"<\|channel>thought\n?", r"<channel\|>\n?",
                                  r"<\|START_THINKING\|>", r"<\|END_THINKING\|>")]

_MARKERS = ("<tool_call>", "<|tool_call>", "<|START_ACTION|>")


def has_leaked_markup(content: str) -> bool:
    if not content:
        return False
    return any(m in content for m in _MARKERS) or "</think>" in content \
        or "<|END_THINKING|>" in content or "<unused25>" in content


def repair_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Return (cleaned_content, tool_calls). tool_calls is [] if none leaked.

    Each call: {"id", "type": "function", "function": {"name", "arguments"}}
    with `arguments` as a JSON string (the shape the executor expects).
    """
    if not content:
        return content or "", []
    calls: list[dict] = []

    for block in _QWEN_BLOCK.findall(content):
        p = _parse_qwen_block(block)
        if p:
            calls.append(p)
    for name, payload in _GEMMA_CALL.findall(content):
        args = _loads_or_raw(payload)
        calls.append(_mk(name, args))
    for payload in _COHERE_ACTION.findall(content):
        calls.extend(_parse_cohere(payload))

    cleaned = content
    cleaned = _QWEN_BLOCK.sub("", cleaned)
    cleaned = _GEMMA_CALL.sub("", cleaned)
    cleaned = _COHERE_ACTION.sub("", cleaned)
    cleaned = _strip_thinking(cleaned).strip()

    # Re-number ids after collecting from all shapes.
    for i, c in enumerate(calls):
        c["id"] = f"repair_{i}"
    return cleaned, calls


def _mk(name: str, args) -> dict:
    if not isinstance(args, dict):
        args = {"_raw": args} if args else {}
    return {"id": "repair", "type": "function",
            "function": {"name": str(name).strip(), "arguments": json.dumps(args, ensure_ascii=False)}}


def _loads_or_raw(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s.strip()}


def _parse_qwen_block(block: str) -> dict | None:
    block = block.strip()
    if block.startswith("{"):
        try:
            obj = json.loads(block)
        except Exception:
            return None
        name = obj.get("name") or obj.get("function", {}).get("name")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            args = _loads_or_raw(args)
        return _mk(name, args) if name else None
    m = _FUNC_NAME.search(block)
    if not m:
        return None
    args: dict = {}
    for key, val in _PARAM.findall(block):
        v = val
        if v.startswith("\n"):
            v = v[1:]
        if v.endswith("\n"):
            v = v[:-1]
        args[key] = v
    return _mk(m.group(1), args)


def _parse_cohere(payload: str) -> list[dict]:
    try:
        arr = json.loads(payload.strip())
    except Exception:
        return []
    if isinstance(arr, dict):
        arr = [arr]
    out = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name") or item.get("name")
        args = item.get("parameters") or item.get("arguments") or {}
        if name:
            out.append(_mk(name, args))
    return out


def _strip_thinking(content: str) -> str:
    if not content:
        return content or ""
    out = content
    for pat in _THINK_BLOCKS:
        out = pat.sub("", out)
    # Stray close marker left by streaming → drop reasoning up to it.
    if any(t in out for t in ("</think>", "<|END_THINKING|>", "<unused25>", "<channel|>")):
        out = _THINK_LEAD.sub("", out, count=1)
    for pat in _STRAY:
        out = pat.sub("", out)
    return out
