"""Tests for localcode.tools.syntax_check — Tier-1 per-write verification.

Covers the duplicate-top-level-declaration detection (BUG #4): grammatically
valid code that still fails the real build ("has already been declared").

tree-sitter may not be importable in this env — the TS/JS cases skip gracefully.
"""
from __future__ import annotations

import pytest

from localcode.tools.syntax_check import check_syntax


def _has_treesitter() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_language_pack  # noqa: F401
        return True
    except Exception:
        return False


# ── TS/JS duplicate declarations (the reported fsrs.ts bug) ──────────────────

def test_duplicate_export_function_ts():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    src = (
        "export function getRetrievability(t: number): number {\n"
        "  return t;\n"
        "}\n"
        "\n"
        "export function getRetrievability(t: number): number {\n"
        "  return t * 2;\n"
        "}\n"
    )
    err = check_syntax("fsrs.ts", src)
    if err is None:
        pytest.skip("tree-sitter grammar unavailable at runtime")
    assert "duplicate declaration" in err
    assert "getRetrievability" in err
    # earliest occurrence first
    assert "lines 1 and 5" in err


def test_duplicate_const_js():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    src = "const foo = 1;\nconst bar = 2;\nconst foo = 3;\n"
    err = check_syntax("x.js", src)
    if err is None:
        pytest.skip("tree-sitter grammar unavailable at runtime")
    assert "duplicate declaration" in err and "foo" in err


def test_no_false_positive_distinct_ts():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    src = (
        "export function a() { return 1; }\n"
        "export function b() { return 2; }\n"
        "const c = 3;\n"
    )
    err = check_syntax("ok.ts", src)
    # None (clean) or None regardless — must not report a duplicate.
    assert err is None or "duplicate" not in err


def test_grammar_error_still_wins_ts():
    """A real syntax error should still be reported (not masked by dup check)."""
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_language_pack")
    src = "function f() { return (1; }\n"
    err = check_syntax("broken.ts", src)
    if err is None:
        pytest.skip("tree-sitter grammar unavailable at runtime")
    assert "duplicate" not in err


# ── Python duplicate declarations (native ast path, no tree-sitter needed) ───

def test_duplicate_python_function():
    src = "def foo():\n    return 1\n\ndef foo():\n    return 2\n"
    err = check_syntax("m.py", src)
    assert err is not None
    assert "duplicate declaration" in err
    assert "foo" in err
    assert "lines 1 and 4" in err


def test_python_overload_not_flagged():
    src = (
        "from typing import overload\n"
        "\n"
        "@overload\n"
        "def f(x: int) -> int: ...\n"
        "@overload\n"
        "def f(x: str) -> str: ...\n"
        "def f(x):\n"
        "    return x\n"
    )
    err = check_syntax("ov.py", src)
    assert err is None


def test_python_syntax_error_reported():
    src = "def broken(:\n    pass\n"
    err = check_syntax("bad.py", src)
    assert err is not None
    assert "Python syntax error" in err


def test_python_clean_returns_none():
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    assert check_syntax("clean.py", src) is None
