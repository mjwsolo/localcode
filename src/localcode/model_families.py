"""Model-family adapter layer.

Until this module, every model-specific detail (thinking tokens, tool-call
delimiters, display-strip patterns) was hardcoded to Gemma 4 across
tool_parsing.py, runtime.py, and tui/screens/chat.py. That made
supporting Qwen / DeepSeek / Llama a copy-paste-and-pray exercise and
let inconsistencies creep in — e.g. we'd remember to strip `<unused25>`
in runtime but not in the TUI display.

This module centralises the model-specific bits behind a `FamilyAdapter`
struct, and the runtime looks up the right adapter based on the active
profile. Default family is GEMMA4 so the behaviour of every code path
that was previously hardcoded-to-Gemma remains byte-identical until a
non-Gemma profile is selected.

Scope of what a family adapter covers:
  - Thinking-channel opening/closing markers (the text the runtime uses
    to detect "we're inside a thinking block").
  - Thinking-channel strip patterns for post-hoc cleanup of already-
    decoded text (what runtime._strip_thinking_tokens does).
  - Tool-call delimiter markers (so the parser knows which regex to
    run). Parser dispatch lives in tool_parsing.py — this module just
    describes the delimiters.
  - Display strip patterns for the TUI so the user never sees the
    raw special tokens.

Scope we DO NOT cover here:
  - Chat template (llama-server handles that via --jinja reading the
    GGUF-embedded template).
  - Stop tokens (we rely on tokenizer EOS; no family has needed an
    override yet).
  - Prompt rendering differences — the SYSTEM_PROMPT content itself
    is still one string, and word-level per-family variants would live
    in eval/prompt_variants.py, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ModelFamily(str, Enum):
    """Enumerated model families we adapt for.

    Add a member here when wiring a new family. The string value doubles
    as the identifier in config files and logs.
    """
    GEMMA4 = "gemma4"
    QWEN = "qwen"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"


@dataclass(frozen=True)
class FamilyAdapter:
    """Per-family behaviour the runtime / parser / TUI consult.

    Fields are deliberately compiled regex objects or literal strings,
    not callables, so the adapter stays data-only and easy to diff.
    Parser dispatch still lives in tool_parsing.py — this struct just
    tells it WHICH delimiter patterns to use.
    """
    family: ModelFamily

    # Opening/closing markers for the thinking channel as they appear in
    # the decoded text stream. The runtime uses these to maintain the
    # in_thinking state machine. For Gemma 4 IQ3_S these are both
    # `<unused25>` (same literal opens and closes — confusing but true).
    thinking_open: str
    thinking_close: str

    # Regex patterns that identify residual thinking/channel markup in
    # already-decoded text. `_strip_thinking_tokens` runs each over the
    # text before yielding content to the user. Include the literal
    # open/close markers here too so a single pass handles both.
    strip_patterns: tuple[re.Pattern, ...] = field(default_factory=tuple)

    # Primary + fallback tool-call delimiter regex patterns. tool_parsing
    # dispatches based on family — this tells it which patterns to try.
    # Keep None for families that use the generic JSON-tool-call fallback
    # in tool_parsing.py.
    tool_call_primary: re.Pattern | None = None
    tool_call_alt: re.Pattern | None = None

    # Optional string delimiter used inside tool-call args (Gemma 4 uses
    # `<|"|>…<|"|>` to wrap string values; most families use plain JSON
    # quotes, so this is None for them).
    arg_string_delim: re.Pattern | None = None


# ── Family registry ─────────────────────────────────────────────────
# Each adapter below documents the sources of its patterns so future
# maintainers can tell what's observed vs assumed.

GEMMA4_ADAPTER = FamilyAdapter(
    family=ModelFamily.GEMMA4,
    # <unused25> is the raw decode of the <|channel>/<channel|> tokens
    # at IQ3_S quantisation — same literal for open and close.
    thinking_open="<unused25>",
    thinking_close="<unused25>",
    strip_patterns=(
        re.compile(r"<unused25>"),
        re.compile(r"<\|channel>thought\n?"),
        re.compile(r"<channel\|>\n?"),
    ),
    tool_call_primary=re.compile(
        r"<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL,
    ),
    tool_call_alt=re.compile(
        r"<\|tool_call\>\s*call\s*:\s*(\w+)\s*\{(.*?)\}\s*<\s*tool_call\s*\|>",
        re.DOTALL,
    ),
    arg_string_delim=re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL),
)


# Qwen uses `<think>…</think>` for reasoning channel and standard
# JSON-in-`<tool_call>` for tool calls. Verified against Qwen 2.5 / 3.x
# output; Qwen 3.6 IQ2_M emits tool names with trailing whitespace
# which tool_parsing.py already handles via .strip() on the extracted
# name — no family-specific knob needed for that.
QWEN_ADAPTER = FamilyAdapter(
    family=ModelFamily.QWEN,
    thinking_open="<think>",
    thinking_close="</think>",
    strip_patterns=(
        re.compile(r"<think>"),
        re.compile(r"</think>"),
    ),
    # Qwen tool calls come as `<tool_call>{"name":…,"arguments":{…}}</tool_call>`
    # which the generic JSON_TOOL_CALL_RE in tool_parsing.py already
    # catches — so no family-specific primary regex needed here.
    tool_call_primary=None,
    tool_call_alt=None,
    arg_string_delim=None,
)


# Llama 3/3.1 instruct models emit tool calls as Python-like function
# signatures inside `<|python_tag|>`. Without a deployed Llama GGUF to
# validate against, the adapter below documents the expected pattern
# but keeps the primary regex None — callers fall back to JSON parsing.
LLAMA_ADAPTER = FamilyAdapter(
    family=ModelFamily.LLAMA,
    # Llama 3 doesn't have a native reasoning channel; these values are
    # set to sentinels that will never match, so the thinking state
    # machine stays inactive for Llama.
    thinking_open="__LLAMA_NO_THINKING_OPEN__",
    thinking_close="__LLAMA_NO_THINKING_CLOSE__",
    strip_patterns=(),
    tool_call_primary=None,
    tool_call_alt=None,
    arg_string_delim=None,
)


# DeepSeek R1 emits reasoning inside `<think>…</think>` (same as Qwen)
# and tool calls in JSON inside `<｜tool▁calls▁begin｜>` blocks. Until
# we're running DeepSeek in production we leave the tool-call regex as
# JSON-fallback.
DEEPSEEK_ADAPTER = FamilyAdapter(
    family=ModelFamily.DEEPSEEK,
    thinking_open="<think>",
    thinking_close="</think>",
    strip_patterns=(
        re.compile(r"<think>"),
        re.compile(r"</think>"),
    ),
    tool_call_primary=None,
    tool_call_alt=None,
    arg_string_delim=None,
)


# Cohere's cohere2 / cohere2_moe (Command R7B, North-Mini-Code) family.
# Reasoning is wrapped in `<|START_THINKING|>…<|END_THINKING|>` and tool
# calls in a JSON array inside `<|START_ACTION|>…<|END_ACTION|>` (vLLM
# "cohere_command4" / command-R style), NOT gemma/qwen/llama formats.
# Patterns below are documented from Cohere's published chat template;
# we have no North-Mini-Code GGUF running on this stack yet, so treat
# the tool-call regex as best-effort and validate against real output
# before relying on it. The thinking markers are exact literals from
# the template and safe to strip.
COHERE_ADAPTER = FamilyAdapter(
    family=ModelFamily.COHERE,
    thinking_open="<|START_THINKING|>",
    thinking_close="<|END_THINKING|>",
    strip_patterns=(
        re.compile(r"<\|START_THINKING\|>"),
        re.compile(r"<\|END_THINKING\|>"),
    ),
    # `<|START_ACTION|>[{"tool_name":…,"parameters":{…}}]<|END_ACTION|>`.
    # Captures the JSON array payload; tool_parsing.py still parses the
    # inner JSON. Assumed from the command-R template — unvalidated on
    # this stack, hence kept as a single capturing group rather than the
    # name/args split the Gemma regex uses.
    tool_call_primary=re.compile(
        r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL,
    ),
    tool_call_alt=None,
    arg_string_delim=None,
)


_REGISTRY: dict[ModelFamily, FamilyAdapter] = {
    ModelFamily.GEMMA4: GEMMA4_ADAPTER,
    ModelFamily.QWEN: QWEN_ADAPTER,
    ModelFamily.LLAMA: LLAMA_ADAPTER,
    ModelFamily.DEEPSEEK: DEEPSEEK_ADAPTER,
    ModelFamily.COHERE: COHERE_ADAPTER,
}


# ── Lookup helpers ──────────────────────────────────────────────────


def get_adapter(family: ModelFamily | str | None) -> FamilyAdapter:
    """Return the adapter for a family, defaulting to Gemma 4.

    Accepts an enum, the string value, or None. Unknown strings also
    fall back to Gemma 4 — callers that want strictness should check
    membership before calling. The default-to-Gemma policy keeps all
    prior callsites byte-identical until a family is explicitly set.
    """
    if family is None:
        return GEMMA4_ADAPTER
    if isinstance(family, str):
        try:
            family = ModelFamily(family)
        except ValueError:
            return GEMMA4_ADAPTER
    return _REGISTRY.get(family, GEMMA4_ADAPTER)


def infer_family_from_profile(profile_id: str) -> ModelFamily:
    """Map a profile id (e.g. 'gemma4-26b-laptop', 'qwen36-35b-iq2') to
    a family. Matching is a simple prefix / substring check because the
    profile ids we generate are stable and prefix-named by family.

    Returns GEMMA4 for anything unrecognised — LocalCode shipped Gemma-
    first and that's the safe default for unknown profiles.
    """
    if not profile_id:
        return ModelFamily.GEMMA4
    low = profile_id.lower()
    if low.startswith("qwen") or "qwen" in low:
        return ModelFamily.QWEN
    if low.startswith("llama") or "llama" in low:
        return ModelFamily.LLAMA
    if low.startswith("deepseek") or "deepseek" in low:
        return ModelFamily.DEEPSEEK
    if "cohere" in low or "north" in low or "command" in low:
        return ModelFamily.COHERE
    return ModelFamily.GEMMA4


# Degenerate-collapse artifacts: a known llama.cpp Gemma-4 bug (esp. the
# 26B-A4B MoE) makes the model spew raw <unusedNN> / [multimodal] tokens in a
# loop. None of these are ever legitimate user-facing content for ANY family,
# so we scrub them globally as defense-in-depth (the streaming layer also
# detects the collapse and stops early). See ggml-org/llama.cpp#21516 / #21321.
_COLLAPSE_TOKEN_RE = re.compile(r"<unused\d+>|\[multimodal\]|<eos>")


def strip_thinking_tokens(text: str, family: ModelFamily | None = None) -> str:
    """Apply the family's thinking-strip patterns to `text`.

    Pure function, no state. Safe to call in hot paths — the compiled
    regexes are attached to the adapter at module-import time.
    """
    if not text:
        return text
    adapter = get_adapter(family)
    out = text
    for pat in adapter.strip_patterns:
        out = pat.sub("", out)
    return _COLLAPSE_TOKEN_RE.sub("", out)
