"""Diff renderer — two-column layout with line-number gutter.

agent pattern from `src/components/StructuredDiff.tsx`:
- Right-aligned gutter (line numbers) at fixed width per hunk.
- Content column with `+`/`-` markers in green/red.
- Context lines dim, hunks separated by `…` row.

Extracted from `chat_log.ChatLog._render_diff` so the diff renderer
can be tested in isolation and (later) reused by other surfaces
(e.g. an inline diff preview during edit_file approval prompts).
"""
from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

from rich.text import Text

from ....theme import C

if TYPE_CHECKING:
    from ..chat_log import ChatLog


def _available_content_width(log: "ChatLog", prefix_width: int) -> int:
    """Chars of CONTENT that fit on one row given the diff prefix.

    Terminal width minus: the diff prefix (gutter + separator + marker),
    RichLog padding, and the vertical scrollbar. Floor at 20 so we
    still wrap on tiny terminals rather than returning ≤0.
    """
    try:
        term_w = log.app.size.width
    except Exception:
        term_w = 80
    # Reserve: 1 for scrollbar, 2 for RichLog padding = 3. Prefix on
    # the left is already accounted for separately.
    return max(20, term_w - prefix_width - 3)


def _emit_wrapped_line(
    log: "ChatLog",
    content: str,
    build_first_prefix,      # callable() → Text for first visual row
    build_cont_prefix,       # callable() → Text for every wrap-continuation row
    content_style: str,
    content_width: int,
    highlighted=None,        # optional pre-syntax-highlighted Text for `content`
) -> None:
    """Write a styled diff line to `log`, wrapping long content cleanly.

    Without this, diff lines longer than the terminal width fell through
    to RichLog's default word-wrap, which rendered the wrapped remainder
    flush to column 0 — text appeared to "extend to the left past the
    margin" (image 110). The fix: pre-split the content on terminal
    width, write the first piece with the real line-number + marker
    prefix, and write each continuation piece with a blank prefix of
    the same width so everything lines up under the `+`/`-` column.

    When `highlighted` (a syntax-colored Text of `content`) is given AND the
    line fits on one row, we render it directly to preserve token colors;
    long lines fall back to plain-style wrapping so layout never breaks.
    """
    # Syntax-highlighted fast path: only when the whole line fits one row
    # (avoids re-implementing style-preserving wrap). Otherwise fall through.
    if highlighted is not None and content_width > 0 and len(content) <= content_width:
        dl = build_first_prefix()
        dl.append_text(highlighted)
        log.write(dl)
        log._track_lines()
        return

    if content_width <= 0:
        # Degenerate terminal size — fall back to one write and let
        # Rich wrap as best it can.
        dl = build_first_prefix()
        dl.append(content, style=content_style)
        log.write(dl)
        log._track_lines()
        return

    # textwrap preserves leading whitespace (important for indented code
    # diffs) but `drop_whitespace=False` also keeps trailing spaces,
    # which is what we want for code.
    pieces = textwrap.wrap(
        content if content else " ",
        width=content_width,
        drop_whitespace=False,
        replace_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
        expand_tabs=False,
    ) or [content]

    dl = build_first_prefix()
    dl.append(pieces[0], style=content_style)
    log.write(dl)
    log._track_lines()

    for extra in pieces[1:]:
        cont = build_cont_prefix()
        cont.append(extra, style=content_style)
        log.write(cont)
        log._track_lines()


_HUNK_HEADER_RE = re.compile(r'\+(\d+)(?:,(\d+))?')
_HUNK_OLD_RE = re.compile(r'-(\d+)')

# Extension → Pygments lexer name for syntax-highlighting diff content.
_LEXER_BY_EXT = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".kt": "kotlin", ".swift": "swift",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".css": "css",
    ".scss": "scss", ".html": "html", ".xml": "xml", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".sh": "bash",
    ".bash": "bash", ".zsh": "bash", ".sql": "sql", ".md": "markdown",
}


def _lexer_for(file_path: str) -> str | None:
    import os
    return _LEXER_BY_EXT.get(os.path.splitext(file_path or "")[1].lower())


def _highlight(content: str, lexer: str | None):
    """Syntax-highlight one code line into a Rich Text, or None on failure.

    Bounded to the few preview lines a diff card shows, so the Pygments cost
    is negligible. Uses `background_color="default"` so it stays transparent
    over the terminal palette (ansi_default theme). Any failure → None, and
    the caller falls back to the flat green/red/dim styling.
    """
    if not lexer or not content.strip():
        return None
    try:
        from rich.syntax import Syntax
        txt = Syntax(
            content, lexer, theme="ansi_dark",
            background_color="default", word_wrap=False,
        ).highlight(content)
        txt.rstrip()  # drop the trailing newline Syntax appends
        return txt
    except Exception:
        return None


def render_diff(log: "ChatLog", diff_text: str, max_body_lines: int = 8) -> None:
    """Write a unified-diff blob to `log` with the structured layout.

    The header row shows file path + change counts (+N -M). The body
    shows up to `max_body_lines` lines of actual diff content; the
    rest is summarized as "… +X more lines".
    """
    lines = diff_text.strip().splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))

    file_path = ""
    max_line_no = 1
    for l in lines:
        if l.startswith("+++ "):
            file_path = l[4:].split("\t", 1)[0]
        elif l.startswith("@@"):
            m = _HUNK_HEADER_RE.search(l)
            if m:
                start = int(m.group(1))
                span = int(m.group(2) or 0)
                max_line_no = max(max_line_no, start + span)
    gutter_w = max(2, len(str(max_line_no)) + 1)
    lexer = _lexer_for(file_path)

    # Header row
    header = Text()
    header.append("    ⎿ ", style=f"dim {C.primary}")
    if file_path:
        header.append(f"{file_path}  ", style=f"dim {C.primary}")
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    header.append(" ".join(parts) if parts else "no changes", style="dim")
    log.write(header)
    log._track_lines()

    # Prefix width: 4 (left pad) + gutter_w + 1 (space) + 2 ("│ " or
    # "│   ") + 2 ("+ " / "- " / "  "). For context lines the "│   "
    # is wider (4 instead of 2) — we compute per-line below. The
    # continuation indent for wrapped long diff lines is a blank
    # string of the same visual width as the prefix, so wrapped text
    # visually hangs under the "+ …" / "- …" column instead of
    # flush-lefting (image 110 bug).
    prefix_w_changed = 4 + gutter_w + 1 + 2 + 2  # "- "/"+ " variant
    prefix_w_context = 4 + gutter_w + 1 + 4      # "│   " + no marker
    cont_indent_changed = " " * prefix_w_changed
    cont_indent_context = " " * prefix_w_context

    width_changed = _available_content_width(log, prefix_w_changed)
    width_context = _available_content_width(log, prefix_w_context)

    old_line = 0
    new_line = 0
    shown = 0
    skipped = 0
    for line_text in lines:
        if line_text.startswith("---") or line_text.startswith("+++"):
            continue
        if line_text.startswith("@@"):
            m = _HUNK_OLD_RE.search(line_text)
            if m:
                old_line = int(m.group(1))
            m2 = _HUNK_HEADER_RE.search(line_text)
            if m2:
                new_line = int(m2.group(1))
            if shown > 0 and shown < max_body_lines:
                sep = Text()
                sep.append(" " * (6 + gutter_w) + "…", style="dim")
                log.write(sep)
                log._track_lines()
                shown += 1
            continue
        if shown >= max_body_lines:
            skipped += 1
            continue

        if line_text.startswith("-"):
            def _first_minus(ol=old_line):
                t = Text()
                t.append(f"    {ol:>{gutter_w}} ", style=f"dim {C.error}")
                t.append("│ ", style=f"dim {C.primary}")
                t.append("- ", style=f"bold {C.error}")
                return t
            def _cont_minus():
                return Text(cont_indent_changed)
            _emit_wrapped_line(
                log, line_text[1:], _first_minus, _cont_minus,
                content_style=f"{C.error}", content_width=width_changed,
            )
            old_line += 1
        elif line_text.startswith("+"):
            def _first_plus(nl=new_line):
                t = Text()
                t.append(f"    {nl:>{gutter_w}} ", style=f"dim {C.success}")
                t.append("│ ", style=f"dim {C.primary}")
                t.append("+ ", style=f"bold {C.success}")
                return t
            def _cont_plus():
                return Text(cont_indent_changed)
            _emit_wrapped_line(
                log, line_text[1:], _first_plus, _cont_plus,
                content_style=f"{C.success}", content_width=width_changed,
                highlighted=_highlight(line_text[1:], lexer),
            )
            new_line += 1
        else:
            content = line_text[1:] if line_text.startswith(" ") else line_text
            if not content.strip():
                old_line += 1
                new_line += 1
                continue
            def _first_ctx(nl=new_line):
                t = Text()
                t.append(f"    {nl:>{gutter_w}} ", style="dim")
                t.append("│   ", style=f"dim {C.primary}")
                return t
            def _cont_ctx():
                return Text(cont_indent_context)
            _emit_wrapped_line(
                log, content, _first_ctx, _cont_ctx,
                content_style="dim", content_width=width_context,
                highlighted=_highlight(content, lexer),
            )
            old_line += 1
            new_line += 1

        shown += 1

    if skipped > 0:
        more = Text()
        more.append(" " * (6 + gutter_w), style="")
        more.append(f"… +{skipped} more lines", style="dim italic")
        log.write(more)
        log._track_lines()
