# Task 03 — Refactor

**Difficulty:** Medium
**Soft budget:** 10 min
**Hard timeout:** 25 min

## Goal (paste into localcode verbatim)

> The function `process_order(order)` in `orders.py` does three things: validates the order, calculates the total, and formats the receipt. Refactor it into three separate functions (`validate_order`, `calculate_total`, `format_receipt`) and have `process_order` call them in sequence. All existing tests in `test_orders.py` must continue to pass without any changes to the test file.

## Success criteria

- `orders.py` defines all four functions: `validate_order`, `calculate_total`, `format_receipt`, `process_order`
- `process_order` still has the same external behavior (same inputs → same outputs)
- `test_orders.py` is unchanged (byte-for-byte) and all tests pass
- `process_order` is now shorter (calls the three helpers, doesn't inline the logic)

## Why this task

Refactoring without breaking tests is a core agent skill. Many local models fail this by either changing the test file (cheating) or breaking the external contract.
