# Task 02 — Fix a bug

**Difficulty:** Small
**Soft budget:** 5 min
**Hard timeout:** 15 min

## Goal (paste into localcode verbatim)

> The file `calculator.py` has a bug: `divide(10, 0)` raises an unhandled `ZeroDivisionError`. Fix it so that `divide(a, 0)` returns `None` and the existing tests still pass. Then add a new test asserting `divide(10, 0) is None`.

## Success criteria

- `calculator.py` still exists and is importable
- `divide(10, 0)` returns `None` (does not raise)
- `divide(10, 2)` still returns `5`
- All existing tests in `test_calculator.py` still pass
- At least one new test in `test_calculator.py` asserts `divide(10, 0) is None`

## Why this task

Tests reading-comprehension on existing code + targeted edit + adding a regression test. The classic "small bug fix" loop.

## Fixtures

`fixtures/calculator.py` and `fixtures/test_calculator.py` are copied into the working dir by `setup.sh`.
