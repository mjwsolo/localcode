# Task 04 — Add a test

**Difficulty:** Medium
**Soft budget:** 10 min
**Hard timeout:** 25 min

## Goal (paste into localcode verbatim)

> Read `parser.py` and write a new pytest file `test_parser.py` that tests `parse_kv_string`. Cover: empty string, single pair, multiple pairs, whitespace tolerance, and at least one malformed input (verify it raises `ValueError`). Aim for at least 5 test functions. Do not modify `parser.py`.

## Success criteria

- `test_parser.py` exists in the working dir
- Contains at least 5 `def test_` functions
- All tests pass: `pytest test_parser.py` exits 0
- `parser.py` is byte-for-byte unchanged

## Why this task

Tests reading comprehension (must understand `parse_kv_string` from source) and the ability to think of edge cases without being told. A common local-model failure is writing one trivial test instead of covering the surface.
