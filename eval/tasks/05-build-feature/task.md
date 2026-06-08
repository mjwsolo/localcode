# Task 05 — Build a feature

**Difficulty:** Hard
**Soft budget:** 20 min
**Hard timeout:** 45 min

## Goal (paste into localcode verbatim)

> In `cli.py` there's a simple Typer CLI with one command `greet`. Add a new command `count` that takes a `--file` argument (path to a text file) and prints two lines: `lines: <N>` and `words: <N>`. Then add tests in `test_cli.py` covering: a normal file, an empty file, and a missing file (should exit non-zero). Use Typer's `CliRunner`. All existing tests must still pass.

## Success criteria

- `cli.py` has both `greet` and `count` commands registered on the same Typer app
- `count --file <path>` prints both `lines: N` and `words: N` to stdout
- `count --file <missing>` exits non-zero
- `test_cli.py` has tests for both commands, all pass
- The existing `test_greet` test still passes unmodified

## Why this task

End-to-end feature: read existing CLI structure, add a new command following the same pattern, handle an error path, write tests for happy + sad paths. This is the closest analog to real day-to-day work.

## Fixtures

`fixtures/cli.py` and `fixtures/test_cli.py` are copied into the working dir.
