---
title: Verification
description: How a turn earns the word "done".
---

A local model that says "I've fixed it" is not evidence. localcode's turn ends
on something checkable instead.

## The loop ends when the model stops calling tools

The primary completion signal is behavioural: the model stops issuing tool
calls and returns a final answer. There is no heuristic that guesses "this
looks finished" from the kind of goal you asked for.

A backstop sits behind that: if the model wrote a todo list during the turn and
items are still open, the loop does not treat a bare final message as
completion.

Each turn records a `completion_status` and a `loop_exit_reason`, and both are
visible in the [JSONL event stream](/localcode/reference/jsonl-events).

## Verification plans come from your repo

localcode does not invent a test command. It looks at what your repo actually
contains and builds a plan from it:

| Repo contains | Steps added |
| --- | --- |
| `ruff.toml` / `.ruff.toml` | `ruff check .` |
| `pyproject.toml` / `pytest.ini` / `tox.ini` | `pytest -q` (plus `python -m compileall src` in thorough mode) |
| `package.json` | `npm run lint -- --if-present`, `npm test -- --runInBand` (plus `npm run build -- --if-present` in thorough mode) |
| `Cargo.toml` | `cargo fmt --check`, `cargo test -q` |

If none of those match, it falls back to a single guessed command —
`pytest -q` for a Python project, `npm test -- --runInBand` for a Node one,
`cargo test -q` for Rust.

Steps are de-duplicated, so a repo with both a `pyproject.toml` and a
`pytest.ini` still runs `pytest -q` once.

## A red check is part of the loop, not the end of it

When a verification step fails, the failing output goes back to the model as
context for the next round. That is the recovery path shown on the home page:
edit → check → read the failure → edit again → check green.

## Syntax checking happens before the shell does

Edits are syntax-checked in-process with tree-sitter before a shell command is
ever run, so an obviously broken edit is caught without spending a test run on
it.

## You still read the diff

Verification is evidence that your repo's own checks pass. It is not a review.
`git diff` and [`/undo`](/localcode/guides/undo) are still yours.
