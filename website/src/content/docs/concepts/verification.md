---
title: Verification
description: What localcode enforces before it will call a turn done, and what it leaves to the model.
---

A local model that says "I've fixed it" is not evidence. localcode does two
things about that — and it is worth being precise about which is a hard gate
and which is a nudge, because they are not the same strength.

## What is enforced

### 1. Code-changing turns cannot claim success without recorded evidence

When the turn's goal is classified as a build or an edit **and** code files
changed, the loop checks whether it observed a passing build / test / typecheck
/ lint command during the turn. The evidence is keyed to the changed files'
content hashes and to `PATH` / `NODE_ENV` / `PYTHONPATH`, so a command that
passed *before* the last edit does not count.

If nothing qualifies, localcode replaces the model's closing text with:

> Implementation changes were made, but LocalCode could not record a passing
> build, test, typecheck, or import check for the current file hashes. The task
> remains incomplete rather than claiming success.

That is the honest shape of the guarantee: localcode does not force the check
to pass, and it does not run your test suite on your behalf — it refuses to
report success it cannot evidence. The turn also reports `incomplete` (exit
code `1`) in [`run --json`](/localcode/reference/jsonl-events).

What counts as evidence is deliberately narrow: a build, typecheck, test or
lint command the model ran through `bash`, which exited zero. Starting a dev
server does not count.

### 2. A stop gate that re-runs the project's own typecheck

For build-shaped goals that changed code, localcode runs the project's real
checker itself when the model tries to finish, and feeds the errors back:

| Project | Command run |
| --- | --- |
| `package.json` with a `typecheck` script | that script |
| TypeScript otherwise | `tsc -p <tsconfig> --noEmit` |
| Python with `ruff` installed | `ruff check --select E9,F` (syntax + undefined names) |
| Python without `ruff` | `python -m compileall -q .` |
| Go | `go build ./...` |
| Rust | `cargo check` |

Every one of these is read-only — verification never writes to your repository,
which is why TypeScript is checked with `--noEmit` rather than a build.

While that check is red, the model **cannot** end the turn: the diagnostics are
injected and another round is forced, up to a bounded retry count so an
unfixable project can't spin forever. A check that times out or fails to run is
recorded as *unverified*, never as clean — and running out of retries does not
promote unverified to verified.

Note the gate runs a typecheck, not your tests.

### 3. Syntax checks before anything else

Writes and edits are syntax-checked in-process with tree-sitter as they happen,
so an obviously broken edit is caught in the same round rather than costing a
shell run.

### 4. Open todos block a bare "done"

If the model wrote a todo list during the turn and items are still open, a
closing message alone is not accepted as completion.

## What is not enforced

- **Your test suite.** localcode never runs `pytest` or `npm test` on its own
  initiative. The model decides to run tests; the enforcement above is about
  whether a passing run was *recorded*, not about causing one.
- **Question-shaped and non-code turns.** The evidence gate is scoped to
  build/edit goals that changed code files. A turn that answers a question or
  edits prose ends when the model stops calling tools.
- **Correctness.** A green checker means the project compiles and the commands
  the model chose passed. It is not a review.

So: not every turn ends on passing evidence. Code-changing turns either carry
that evidence or are reported as incomplete.

## You still read the diff

```sh
git diff
```

[`/undo`](/localcode/guides/undo) reverts the agent's file changes if the
answer is no.
