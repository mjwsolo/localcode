"""read_file — read a file with optional line offset/limit."""
from __future__ import annotations

from .base import ToolContext


DEFAULT_LIMIT = 240
MAX_DEFAULT_CHARS = 12_000


def _dynamic_read_defaults(ctx: ToolContext) -> tuple[int, int]:
    """(default_line_limit, max_default_chars), scaled to the model's real
    context window. read_file is the primary code-ingestion path: on a 256K
    window the fixed 240-line / 12K-char default forces needless pagination and
    re-reads where the machine has the MOST room. Scale up to ~6% of the window
    in chars (and lines proportionally), floored at the static defaults so a
    16 GB Mac (small window) is byte-identical to before.
    """
    try:
        ctx_tokens = int(ctx.app.engine._target_num_ctx())
    except Exception:
        ctx_tokens = 0
    if not ctx_tokens:
        return DEFAULT_LIMIT, MAX_DEFAULT_CHARS
    max_chars = max(MAX_DEFAULT_CHARS, int(ctx_tokens * 3.5 * 0.06))
    # Keep the historical ~50 chars/line ratio between the two defaults.
    line_limit = max(DEFAULT_LIMIT, max_chars // 50)
    return line_limit, max_chars

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file. You MUST read before editing. Returns content with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "offset": {"type": "integer", "description": "Start line (0-based). Optional."},
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max lines. Default 240. Use a small targeted range "
                        "for large files; only request a larger limit when the "
                        "whole file is genuinely needed."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


# Fuzzy-match ratio at/above which we AUTO-READ the real file instead of
# merely suggesting it. 0.85 is high enough that only genuine typos /
# case-variants of one real name match (e.g. a dropped letter or wrong
# case), not two distinct files that happen to be somewhat similar.
AUTO_READ_RATIO = 0.85


def _list_dir_candidates(missing):
    """(candidates, base) — every file under the nearest existing ancestor of
    `missing`. Walk up to 6 levels so a wrong path segment deep in the tree
    still resolves to a searchable directory. Capped at 4000 files."""
    from pathlib import Path
    missing = Path(missing)
    base = missing.parent
    for _ in range(6):
        if base.exists():
            break
        base = base.parent
    if not base.exists():
        return [], None
    candidates = []
    for p in base.rglob("*"):
        if p.is_file():
            candidates.append(p)
        if len(candidates) > 4000:
            break
    return candidates, base


def _confident_match(missing):
    """Return the ONE real file `missing` is almost certainly a typo of, else
    None. Authoritative (used to auto-read), so it only fires when unambiguous:

      • exactly one file whose basename equals `missing`'s case-insensitively
        (a pure case/typo of a unique name — gitHub→Github, README.MD→readme.md), OR
      • exactly one file whose basename fuzzy-matches at ratio ≥ AUTO_READ_RATIO.

    Requiring the winner to be UNIQUE prevents auto-reading the wrong one of
    several same-named files in different dirs — that stays a soft suggestion.
    """
    import difflib
    from pathlib import Path
    try:
        candidates, _ = _list_dir_candidates(missing)
        if not candidates:
            return None
        name = Path(missing).name
        # Pure case/typo of a unique basename → definitely that file.
        same = [p for p in candidates if p.name.lower() == name.lower()]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            return None  # ambiguous by name — don't auto-pick.
        # High-confidence fuzzy basename match, and it must be unique.
        names = [p.name for p in candidates]
        close = difflib.get_close_matches(name, names, n=1, cutoff=AUTO_READ_RATIO)
        if close:
            hits = [p for p in candidates if p.name == close[0]]
            if len(hits) == 1:
                return hits[0]
    except Exception:
        pass
    return None


def _suggest_path(missing) -> str:
    """Suggest the real file when the model gave a typo'd/wrong-case path.

    Walk up to the nearest existing ancestor dir and fuzzy-match the missing
    filename (and its path segments) against what's actually there. Lower-
    confidence sibling of `_confident_match`: used only when we're NOT sure
    enough to auto-read, so it stays a hint the model can act on.
    """
    import difflib
    from pathlib import Path
    try:
        missing = Path(missing)
        name = missing.name
        candidates, base = _list_dir_candidates(missing)
        if not candidates:
            return ""
        names = [p.name for p in candidates]
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if close:
            match = next(p for p in candidates if p.name == close[0])
            return f"Did you mean: {match} — read THAT exact path (don't guess variants)."
    except Exception:
        pass
    return ""


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for read_file."
    requested = args["path"]
    path = ctx.resolve_path(requested)
    note = ""
    source_disp = str(requested)
    if not path.exists():
        # A small model invents misspelled/wrong-case paths (a dropped letter,
        # gitHub/github/Github) and, getting a bare "not found", retries with
        # ANOTHER wrong variant forever (dedup can't collapse differing typos).
        real = _confident_match(path)
        if real is None:
            # Not confident enough to auto-read — hand back the best soft hint.
            hint = _suggest_path(path)
            return f"File not found: {requested}" + (f"\n{hint}" if hint else "")
        # AUTHORITATIVE recovery: `requested` is unambiguously a typo/case-
        # variant of exactly one real file. Read THAT file now instead of
        # bouncing the model back with a suggestion it can ignore (which is
        # what fuels the 20+-retry not-found death loop). The note pins the
        # correct path so the next call uses it directly.
        try:
            source_disp = str(real.relative_to(ctx.repo))
        except Exception:
            source_disp = str(real)
        note = (
            f"[read {source_disp} instead of '{requested}' — that path does not "
            f"exist and this is the only close match. Use '{source_disp}' "
            "exactly from now on; do NOT guess more spellings.]\n\n"
        )
        path = real
    if path.is_dir():
        # The model routinely read_file's a directory (confusing it with
        # list_files), gets IsADirectoryError, and burns a whole round before
        # retrying with list_files. Just hand back the listing + a nudge so the
        # round is productive instead of wasted.
        try:
            entries = sorted(
                p.name + ("/" if p.is_dir() else "") for p in path.iterdir()
            )
        except Exception as e:
            return f"Error: '{source_disp}' is a directory and could not be listed: {e}"
        shown = entries[:200]
        more = f"\n  … ({len(entries) - len(shown)} more)" if len(entries) > len(shown) else ""
        body = "\n".join(f"  {e}" for e in shown) or "  (empty)"
        return (
            f"'{source_disp}' is a directory, not a file ({len(entries)} entries). "
            f"read_file needs a FILE path; use list_files for directories. Contents:\n"
            f"{body}{more}"
        )
    content = path.read_text(errors="replace")
    lines = content.splitlines()
    default_limit, max_default_chars = _dynamic_read_defaults(ctx)
    offset = args.get("offset", 0)
    explicit_limit = "limit" in args
    limit = args.get("limit", default_limit)
    # A read is "full" (satisfies the read-before-edit guard) only when it
    # starts at the top AND returns every line untruncated. offset>0 or any
    # truncation below flips this to a partial view.
    _full_read = (not offset) and (len(lines) <= offset + limit)
    selected = lines[offset:offset + limit]
    numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
    result = "\n".join(numbered)
    if len(lines) > offset + limit:
        result += f"\n\n[{len(lines) - offset - limit} more lines — use offset={offset + limit} to continue]"
    if not explicit_limit and len(result) > max_default_chars:
        kept: list[str] = []
        total = 0
        for line in numbered:
            total += len(line) + 1
            if total > max_default_chars:
                break
            kept.append(line)
        result = "\n".join(kept)
        remaining_from_line = offset + len(kept)
        result += (
            f"\n\n[Large file summarized at {max_default_chars} chars. "
            f"File has {len(lines)} lines; continue with "
            f"offset={remaining_from_line}, limit={default_limit}, or request "
            "a focused smaller range around the symbol you need.]"
        )
        _full_read = False  # char-truncated: the model has not seen the whole file
    # Record this read for the read-before-edit staleness guard (edit_file /
    # write_file / multi_edit consult it). Best-effort; never fail a read.
    try:
        from . import read_state
        read_state.record_read(ctx.app, path, full=_full_read)
    except Exception:
        pass
    # Prompt-injection defence: wrap untrusted file content in explicit
    # data/instruction separator markers so the model knows this text
    # is DATA, not commands. Signature detector flags common injection
    # phrases (IGNORE ALL PRIOR INSTRUCTIONS, etc.) with a visible
    # warning before the content. See src/localcode/injection_defense.py.
    from ..injection_defense import wrap_untrusted
    return note + wrap_untrusted(result, source=source_disp)


def is_concurrency_safe(args: dict) -> bool:
    return True
