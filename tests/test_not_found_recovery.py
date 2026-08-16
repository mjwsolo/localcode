"""Authoritative path-not-found recovery — the read-loop breaker.

Two layers:

  1. read_file._confident_match / execute — when a requested path is
     unambiguously a typo/case-variant of exactly one real file, read_file
     AUTO-READS that file (returning its content + a note pinning the real
     path) instead of handing back a soft "did you mean" the model ignores.

  2. recovery.detect_not_found_loop / not_found_nudge — the read-side analogue
     of the bash command-failure churn guard. If read_file keeps failing
     'file not found' against the same parent directory, emit a hard nudge
     telling the model to list_files that dir once and use an exact name.
"""
from __future__ import annotations


# ── read_file auto-read (layer 1) ───────────────────────────────────


class _App:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.session = type("_S", (), {"current_task": None})()


class _Out:
    pass


def _ctx(tmp_path):
    from localcode.tools.base import ToolContext
    return ToolContext(app=_App(tmp_path), out=_Out())


def test_confident_match_resolves_case_variant(tmp_path):
    # gitHub.md → Github.md: fuzzy ratio is below 0.85, but the basename
    # matches case-insensitively and is unique, so _confident_match resolves
    # it authoritatively. (Tested at the function level because macOS's
    # case-insensitive FS would resolve the variant before execute() runs;
    # on case-sensitive Linux CI this is exactly the not-found path.)
    from localcode.tools.read_file import _confident_match
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "Github.md").write_text("the real contents\n")

    match = _confident_match(tmp_path / "notes" / "gitHub.md")

    assert match is not None and match.name == "Github.md"


def test_auto_reads_single_letter_typo(tmp_path):
    # Aki.md → Anki.md: ratio 0.92 ≥ AUTO_READ_RATIO, unique → auto-read.
    from localcode.tools.read_file import execute
    (tmp_path / "Anki.md").write_text("anki deck notes\n")

    out = execute(_ctx(tmp_path), {"path": "Aki.md"})

    assert "anki deck notes" in out
    assert "Anki.md" in out
    assert "read Anki.md instead" in out.replace("./", "")


def test_ambiguous_basename_does_not_auto_read(tmp_path):
    # Two real files share a basename → we must NOT guess which one; fall back
    # to the soft suggestion instead of auto-reading the wrong file.
    from localcode.tools.read_file import execute
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    (tmp_path / "x" / "config.py").write_text("x one\n")
    (tmp_path / "y" / "config.py").write_text("y one\n")

    out = execute(_ctx(tmp_path), {"path": "CONFIG.py"})

    assert out.startswith("File not found")
    assert "x one" not in out and "y one" not in out


def test_no_close_match_returns_plain_not_found(tmp_path):
    from localcode.tools.read_file import execute
    (tmp_path / "readme.txt").write_text("hi\n")

    out = execute(_ctx(tmp_path), {"path": "zzzzzzzz.md"})

    assert out.startswith("File not found: zzzzzzzz.md")


def test_real_file_reads_unchanged(tmp_path):
    # A correct path is byte-identical to before — no note, no auto-read path.
    from localcode.tools.read_file import execute
    (tmp_path / "main.py").write_text("print('hi')\n")

    out = execute(_ctx(tmp_path), {"path": "main.py"})

    assert "print('hi')" in out
    assert "instead of" not in out


def test_read_file_on_directory_returns_listing_not_error(tmp_path):
    # read_file on a directory should return its listing + a nudge, not error —
    # so the model doesn't burn a round before retrying with list_files.
    from localcode.tools.read_file import execute
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("x")

    out = execute(_ctx(tmp_path), {"path": "."})

    assert "is a directory" in out
    assert "a.py" in out and "sub/" in out
    assert not out.startswith("Error")
    assert "list_files" in out


# ── recovery.detect_not_found_loop (layer 2) ────────────────────────


from localcode.agent.recovery import (
    NOT_FOUND_LOOP_LIMIT,
    not_found_key,
    detect_not_found_loop,
    not_found_nudge,
)


def test_not_found_key_collapses_variants_by_parent():
    # Differing typos of one filename share a parent → one count family.
    assert not_found_key("notes/Aki.md") == "notes"
    assert not_found_key("notes/Anki.md") == "notes"
    assert not_found_key("notes/anki.md") == "notes"
    assert not_found_key("bare.md") == "."
    # Backslashes normalized so Windows-style paths key the same.
    assert not_found_key("a\\b\\c.md") == "a/b"


def test_loop_fires_at_threshold():
    hit = detect_not_found_loop({"notes": NOT_FOUND_LOOP_LIMIT})
    assert hit is not None
    parent, count = hit
    assert parent == "notes" and count == NOT_FOUND_LOOP_LIMIT


def test_loop_below_threshold_does_not_fire():
    assert detect_not_found_loop({"notes": NOT_FOUND_LOOP_LIMIT - 1}) is None
    assert detect_not_found_loop({}) is None


def test_loop_reports_worst_directory():
    hit = detect_not_found_loop(
        {"docs": 1, "notes": NOT_FOUND_LOOP_LIMIT + 2, "src": 2}
    )
    assert hit == ("notes", NOT_FOUND_LOOP_LIMIT + 2)


def test_nudge_is_forward_only_and_pins_recovery():
    text = not_found_nudge("notes")
    assert "notes" in text
    assert "list_files" in text
    # Forward-only imperative — no echo-able self-critical loop language a
    # small model would parrot back into its own context.
    for banned in ("circles", "keep failing", "you keep", "giving up", "stop guessing"):
        assert banned.lower() not in text.lower()


def test_nudge_pins_exact_real_names_when_available():
    text = not_found_nudge("notes", real_names=["Anki.md", "todo.md"])
    assert "Anki.md" in text and "todo.md" in text
    assert "EXACT" in text
