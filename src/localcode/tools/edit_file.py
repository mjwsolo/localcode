"""edit_file — replace exact text in a file with small-model robustness.

Designed for IQ2/IQ3-quantized models that routinely make small mistakes
in `old_string`: extra/missing whitespace, included Read-output line
number prefixes, off-by-one indentation. Without robustness these
mistakes look like "old_string not found" → the model gives up and
falls back to write_file (full rewrite) — the failure mode the user
keeps hitting in practice.

Four-tier matching cascade
--------------------------
1. EXACT match (current behavior).
2. STRIP READ PREFIX: if `old_string` starts with `<digits>\\t<rest>` on
   each line, strip the prefix and retry exact. Auto-handles the
   model copying line numbers from `read_file` output.
3. WHITESPACE NORMALISED: collapse runs of whitespace inside both
   `old_string` and the file content; if the normalised match is
   UNIQUE, accept it.
4. CLOSEST MATCH SUGGESTIONS: when nothing matches, return up to 3
   nearest matches with line numbers so the model can self-correct
   on the next try instead of giving up.

Plus `replace_all` flag (agent parity) for renames.

Safety
------
- Whitespace-normalised match is only accepted when UNIQUE, otherwise
  ambiguous. We never silently pick "the first one" for a fuzzy match.
- Stripping the line-prefix only fires when the pattern matches on
  EVERY line of `old_string` — partial matches don't trigger.
- `replace_all` requires the flag to be explicitly set; default is
  single-replacement, same as today, so existing callers don't change.
"""
from __future__ import annotations

import difflib
import re

from .base import ToolContext


SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Replace text in a file. ALWAYS prefer this over write_file when "
            "modifying an existing file — write_file replaces the entire file "
            "and burns context. Read the file first. Keep `old_string` SMALL "
            "but UNIQUE — usually 2-4 adjacent lines is plenty. Set "
            "`replace_all: true` to rename across all occurrences. Whitespace "
            "is matched flexibly: minor indentation/spacing differences in "
            "`old_string` will still find the match if it's unique. Don't "
            "include the `<line>\\t` prefix from read_file output in "
            "`old_string` — write the actual file content only. Do not include "
            "large unchanged regions; if the replacement is truly huge or "
            "repetitive, prefer focused edits or generate repetitive content "
            "locally."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {
                    "type": "string",
                    "description": "Text to find (small + unique, 2-4 lines is ideal)",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace EVERY occurrence of old_string. Default false. "
                        "Use for renames (variable, function, type)."
                    ),
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}


# Line-number prefix patterns from `read_file` output. We see two:
#  • compact form: "  42\t<content>"  (leading-spaces + digits + tab)
#  • cat-n form:  "    42  <content>" (cat -n style, 6 chars then content)
_PREFIX_PATTERNS = [
    re.compile(r"^\s*\d+\t"),
    re.compile(r"^\s*\d+\s{2}"),
]


def _strip_read_prefix(text: str) -> str | None:
    """If EVERY non-empty line starts with a Read-style line prefix,
    strip it and return the cleaned text. Returns None if the pattern
    doesn't match all lines (so we don't accidentally strip from real
    content that happens to start with a digit + tab)."""
    lines = text.splitlines(keepends=True)
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return None
    for pat in _PREFIX_PATTERNS:
        if all(pat.match(l) for l in non_empty):
            return "".join(pat.sub("", l) for l in lines)
    return None


# Typography normalisation — length-preserving char→char map. A small model
# (or a copy-paste from rendered markdown / a chat UI) routinely emits STRAIGHT
# quotes where the file has CURLY ones (or vice-versa), ASCII hyphens where the
# file has en/em dashes, and normal spaces where the file has NBSP. Normalising
# both sides to ASCII turns a hard "not found" into a correct edit. Because the
# map is 1:1 (every char maps to exactly ONE char) the normalised string is the
# SAME LENGTH as the original, so a match index in normalised space points at
# the identical index in the original — we recover the real bytes for free.
# (claude-code utils.ts normalizeQuotes; codex seek_sequence.rs normalise;
# pi edit-diff.ts normalizeForFuzzyMatch.)
_TYPO_MAP = {
    ord(k): v
    for k, v in {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single curly
        "“": '"', "”": '"', "„": '"', "‟": '"',  # double curly
        "‐": "-", "‑": "-", "‒": "-", "–": "-",  # dashes
        "—": "-", "―": "-", "−": "-",
        " ": " ", " ": " ", " ": " ", " ": " ",  # odd spaces
        " ": " ", " ": " ", " ": " ", " ": " ",
        " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    }.items()
}


def _normalise_typography(text: str) -> str:
    """Length-preserving map of curly quotes / unicode dashes / odd spaces to
    their ASCII equivalents."""
    return text.translate(_TYPO_MAP)


def _is_opening_quote(s: str, i: int) -> bool:
    """Opening context (claude-code isOpeningContext): start of string or
    preceded by whitespace / an opening bracket / a dash."""
    if i == 0:
        return True
    return s[i - 1] in " \t\n\r([{—–"


def _preserve_typography(actual_old: str, new: str) -> str:
    """Re-apply the FILE's smart-quote style to `new` (claude-code
    preserveQuoteStyle). If the matched file text used curly quotes, convert the
    corresponding straight quotes in the replacement back to curly so we don't
    flatten the file's typography. Only quotes are re-applied (dashes/spaces are
    left as the model wrote them — matching claude-code)."""
    has_curly_double = ("“" in actual_old) or ("”" in actual_old)
    has_curly_single = ("‘" in actual_old) or ("’" in actual_old)
    if not (has_curly_double or has_curly_single):
        return new
    out: list[str] = []
    for i, ch in enumerate(new):
        if ch == '"' and has_curly_double:
            out.append("“" if _is_opening_quote(new, i) else "”")
        elif ch == "'" and has_curly_single:
            # A quote between two letters is a contraction (don't → don’t).
            if 0 < i < len(new) - 1 and new[i - 1].isalpha() and new[i + 1].isalpha():
                out.append("’")
            else:
                out.append("‘" if _is_opening_quote(new, i) else "’")
        else:
            out.append(ch)
    return "".join(out)


def _find_typography_match(content: str, old: str) -> tuple[int, int] | None:
    """If `old` matches `content` UNIQUELY after typography normalisation,
    return (start, end) of the ORIGINAL (real-bytes) span. Length-preserving
    normalisation means the normalised match index equals the original index."""
    norm_old = _normalise_typography(old)
    norm_content = _normalise_typography(content)
    # If typography normalisation changes NEITHER side, this is identical to the
    # exact tier (which already ran and failed) — nothing new to try. Otherwise
    # a difference exists on the old side, the file side, or both, so proceed.
    if norm_old == old and norm_content == content:
        return None
    hits: list[int] = []
    start = 0
    while True:
        idx = norm_content.find(norm_old, start)
        if idx < 0:
            break
        hits.append(idx)
        start = idx + 1
    if len(hits) != 1:
        return None
    return (hits[0], hits[0] + len(old))


_WS_RUN = re.compile(r"\s+")


def _normalise_ws(text: str) -> str:
    """Collapse runs of whitespace to a single space + strip ends, after
    typography normalisation so a match differing by BOTH quote style and
    whitespace still lands. Used for fuzzy match: we don't accept fuzzy matches
    blindly, only when the normalised pattern occurs UNIQUELY in the normalised
    file."""
    return _WS_RUN.sub(" ", _normalise_typography(text)).strip()


def _find_normalised_match(content: str, old: str) -> tuple[int, int] | None:
    """If `old` matches `content` uniquely after whitespace normalisation,
    return (start_in_content, end_in_content) of the original match.
    Otherwise None.

    Approach: build a position map from normalised offsets back to
    original offsets, search for the normalised needle, then translate
    the unique hit back to the original substring boundaries.
    """
    norm_old = _normalise_ws(old)
    if not norm_old:
        return None
    # Build (normalised_text, normalised_pos -> original_pos) map. We normalise
    # typography first (length-preserving, so index i still maps to content[i])
    # then collapse whitespace runs, exactly matching `_normalise_ws(old)`.
    typo_content = _normalise_typography(content)
    norm_chars: list[str] = []
    pos_map: list[int] = []
    in_ws = False
    for i, ch in enumerate(typo_content):
        if ch.isspace():
            if not in_ws:
                norm_chars.append(" ")
                pos_map.append(i)
                in_ws = True
            # consecutive whitespace: collapse, don't record position
        else:
            norm_chars.append(ch)
            pos_map.append(i)
            in_ws = False
    norm_text = "".join(norm_chars).strip()
    # We stripped leading/trailing — adjust pos_map to match. Simplest:
    # find the offset of the stripped region inside the unstripped form.
    raw_norm = "".join(norm_chars)
    lstrip = len(raw_norm) - len(raw_norm.lstrip())
    pos_map = pos_map[lstrip:lstrip + len(norm_text)]

    # Find all occurrences of norm_old in norm_text — must be unique.
    hits = []
    start = 0
    while True:
        idx = norm_text.find(norm_old, start)
        if idx < 0:
            break
        hits.append(idx)
        start = idx + 1
    if len(hits) != 1:
        return None
    norm_start = hits[0]
    norm_end = norm_start + len(norm_old)
    # Translate back to original offsets. End is the position AFTER the
    # last normalised char of the match — bound to length of content.
    orig_start = pos_map[norm_start] if norm_start < len(pos_map) else len(content)
    orig_end_inclusive = pos_map[norm_end - 1] if (norm_end - 1) < len(pos_map) else len(content) - 1
    # Extend end to capture a trailing whitespace run that was normalised
    # away (so the replacement doesn't strand a blank).
    orig_end = orig_end_inclusive + 1
    while orig_end < len(content) and content[orig_end].isspace():
        # Stop at newline boundary so we don't gobble structural spacing
        if content[orig_end] == "\n":
            break
        orig_end += 1
    return (orig_start, orig_end)


def _closest_matches(content: str, old: str, k: int = 3) -> list[tuple[int, str]]:
    """Return up to k (line_number, snippet) tuples — the lines whose
    content is closest to `old`'s first non-empty line. Used to give
    the model a usable error message when no match is found, so it
    can adjust its next call instead of giving up.
    """
    needle = ""
    for line in old.splitlines():
        if line.strip():
            needle = line.strip()
            break
    if not needle:
        return []
    lines = content.splitlines()
    scored: list[tuple[float, int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        ratio = difflib.SequenceMatcher(None, needle, s).ratio()
        if ratio > 0.4:    # ignore wildly unrelated lines
            scored.append((ratio, i, line))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(i + 1, line) for _, i, line in scored[:k]]


def _record_write(ctx: ToolContext, path, content: str) -> None:
    """Refresh read-state after our own write so the model's own edits never
    trip the read-before-edit guard next round. Best-effort."""
    try:
        from . import read_state
        read_state.record_write(ctx.app, path, content)
    except Exception:
        pass


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for edit_file."
    path = ctx.resolve_path(args["path"])
    if not path.exists():
        return f"File not found: {args['path']}"

    # ── Read-before-edit staleness guard ──
    # Refuse to edit a file the model hasn't fully read this session, or that
    # has changed on disk since it was read. See tools/read_state.py.
    try:
        from . import read_state
        _guard = read_state.guard_edit(ctx.app, path, args["path"])
    except Exception:
        _guard = None
    if _guard:
        return _guard

    # Guard required args BEFORE dereferencing them — a small model that omits
    # `new_string` (or `old_string`) must get a clear, recoverable message, not
    # an unhandled KeyError that reads as a tool crash.
    if "old_string" not in args:
        return ("Error: 'old_string' is required for edit_file — the exact text "
                "to find. To create or fully rewrite a file, use write_file.")
    if "new_string" not in args:
        return ("Error: 'new_string' is required for edit_file — the replacement "
                "text. Use an empty string \"\" to delete the matched text.")
    content = path.read_text(errors="replace")
    old = args["old_string"]
    new = args["new_string"]
    replace_all = bool(args.get("replace_all", False))

    if not old:
        return "Error: old_string is empty — nothing to find/replace."

    # ── Tier 1: exact match ──
    occurrences = content.count(old)
    if occurrences > 0 and not replace_all and occurrences > 1:
        # Multiple exact matches: ambiguous. Tell the model how to fix.
        return (
            f"old_string matches {occurrences} places in {args['path']}. "
            f"Either expand `old_string` with more surrounding context to "
            f"make it unique, or set `replace_all: true` if you want to "
            f"change every occurrence (rename pattern)."
        )
    if occurrences >= 1:
        old_content = content
        if replace_all:
            new_content = content.replace(old, new)
            count = occurrences
        else:
            new_content = content.replace(old, new, 1)
            count = 1
        # No-op detector. If old_string and new_string produce identical
        # bytes (most commonly: model emitted old==new, or new is already
        # what's at that position so the replace is a self-replace), the
        # file doesn't change. Returning "Edited (1 replacement)" in that
        # case lies to the model — it thinks the edit landed and re-emits
        # the same edit, infinite loop. Real failure 2026-04-26: model
        # called edit_file 12+ times on the same path with old==new
        # because each call returned a success string. Surface the no-op
        # as an explicit error so the model picks a different approach.
        if new_content == old_content:
            return (
                f"Error: no-op edit on {args['path']}. The replacement "
                f"did not change any bytes — `old_string` and `new_string` "
                f"produce identical content. Common causes: (1) `old_string` "
                f"and `new_string` are the same; (2) the change you intended "
                f"is already in the file. Re-read the file and either pick "
                f"a different `old_string`/`new_string` pair, or stop — the "
                f"file may already be in the state you want."
            )
        path.write_text(new_content)
        _record_write(ctx, path, new_content)
        _sw = ""
        try:
            from .syntax_check import check_syntax
            _e = check_syntax(str(path), new_content)
            if _e:
                _sw = f"\n\n⚠ SYNTAX ERROR after this edit — fix it now: {_e}"
        except Exception:
            _sw = ""
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=args["path"], tofile=args["path"], lineterm="",
        ))
        if diff:
            return "\n".join(diff[:60]) + _sw
        return f"Edited {args['path']} ({count} replacement{'s' if count != 1 else ''}){_sw}"

    # ── Tier 2: strip Read-output line-number prefixes from old_string ──
    stripped = _strip_read_prefix(old)
    if stripped is not None and stripped and stripped in content:
        # Recursive call with cleaned old_string. Safe: stripped never
        # has the prefix pattern, so it can't recurse again.
        return execute(ctx, {**args, "old_string": stripped})

    # ── Tier 3: typography-normalised match (curly/straight quotes, unicode
    # dashes, odd spaces) — unique only. Preserves the FILE's typography by
    # re-applying its quote style to new_string (claude-code preserveQuoteStyle).
    typo_span = _find_typography_match(content, old)
    if typo_span is not None:
        start, end = typo_span
        old_content = content
        actual_old = content[start:end]
        new_adj = _preserve_typography(actual_old, new)
        new_content = content[:start] + new_adj + content[end:]
        if new_content != old_content:
            path.write_text(new_content)
            _record_write(ctx, path, new_content)
            diff = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=args["path"], tofile=args["path"], lineterm="",
            ))
            body = "\n".join(diff[:60]) if diff else (
                f"Edited {args['path']} (typography-tolerant match)"
            )
            return (
                f"NOTE: matched `old_string` after normalising quote/dash "
                f"typography (straight vs curly quotes, unicode dashes). Edit "
                f"applied successfully, preserving the file's typography.\n{body}"
            )

    # ── Tier 4: whitespace-normalised match (unique only) ──
    span = _find_normalised_match(content, old)
    if span is not None:
        start, end = span
        old_content = content
        actual_old = content[start:end]
        # Build new content, only the unique fuzzy-match span gets replaced.
        new_content = content[:start] + new + content[end:]
        path.write_text(new_content)
        _record_write(ctx, path, new_content)
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=args["path"], tofile=args["path"], lineterm="",
        ))
        body = "\n".join(diff[:60]) if diff else (
            f"Edited {args['path']} (whitespace-tolerant match)"
        )
        return (
            f"NOTE: matched `old_string` after whitespace normalisation — "
            f"your `old_string` had {len(old) - len(actual_old)} char "
            f"difference vs the file. Edit applied successfully.\n{body}"
        )

    # ── Tier 5: nothing matched — return actionable suggestions ──
    near = _closest_matches(content, old, k=3)
    if near:
        lines_summary = "\n".join(
            f"  line {ln}: {snip[:120]}" for ln, snip in near
        )
        return (
            f"old_string not found in {args['path']}. The {len(near)} closest "
            f"line(s) in the file:\n{lines_summary}\n\n"
            f"Re-read the file with read_file to see the EXACT current text, "
            f"then call edit_file again with old_string copied verbatim. Do "
            f"NOT include the leading '<digits>\\t' prefix that read_file "
            f"shows — that's display formatting, not part of the file."
        )
    return (
        f"old_string not found in {args['path']}. No close matches either — "
        f"the file may have changed since you read it. Re-read with read_file "
        f"and try again with the current text."
    )
