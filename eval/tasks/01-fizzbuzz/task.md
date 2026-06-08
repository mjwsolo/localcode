# Task 01 — FizzBuzz

**Difficulty:** Trivial
**Soft budget:** 2 min
**Hard timeout:** 5 min

## Goal (paste into localcode verbatim)

> Write a Python script `fizzbuzz.py` that prints the numbers 1–100, replacing multiples of 3 with "Fizz", multiples of 5 with "Buzz", and multiples of both with "FizzBuzz". Then add a `pytest`-style test `test_fizzbuzz.py` that imports the function and asserts the output for n=15.

## Success criteria

- `fizzbuzz.py` exists in the working dir
- Running `python fizzbuzz.py` produces output where line 3 is `Fizz`, line 5 is `Buzz`, line 15 is `FizzBuzz`
- `test_fizzbuzz.py` exists, contains at least one `def test_` function
- `pytest test_fizzbuzz.py` exits 0

## Why this task

Sanity check that the agent loop runs end-to-end against the local model: can it write a small file, can it write a test, does the test pass. If this fails, nothing else matters.
