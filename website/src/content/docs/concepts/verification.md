---
title: Verification
description: What localcode enforces before it will call a turn done, and what it leaves to the model.
---

A local model that says "I've fixed it" is not evidence. localcode does two
things about that — and it is worth being precise about which is a hard gate
and which is a nudge, because they are not the same strength.

## What is enforced

### 1. Build and edit turns cannot claim success without recorded evidence

The scope is exact, and worth knowing before you rely on it. localcode
classifies each request with a regex-based goal classifier
(`infer_goal_state`). The evidence gate applies to exactly two of its outputs:

- **`build_app`** — a build verb (build, create, make, scaffold, implement,
  write, generate…) together with an app-ish noun (app, site, api, cli, tool,
  library…).
- **`edit_existing`** — one of `fix`, `change`, `edit`, `update`, `refactor`,
  `rename`, `remove`, `add`.

Anything else lands in `general_task`, `run_or_launch` or `question`, and is
**not** covered. Phrasing matters more than it should: *"change parse_duration
so it accepts 1h30m"* is gated; *"make parse_duration accept 1h30m"* is not.

Within that scope, and only when code files changed, the loop checks whether it
observed a passing build / test / typecheck / lint command during the turn. The evidence is keyed to the changed files'
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

For `build_app` goals that changed code, localcode runs a checker itself when
the model tries to finish, and feeds the errors back:

| Project | Command run |
| --- | --- |
| TypeScript (`tsconfig.json` + installed `tsc`, deps present) | `node_modules/.bin/tsc -p <tsconfig> --noEmit` |
| Python with `ruff` installed | `ruff check --select E9,F` (syntax + undefined names) |
| Python without `ruff` | `python -m compileall -q .` |
| Go | `go build ./...` |
| Rust | `cargo check` |

It runs the **`tsc` binary directly** and deliberately does *not* run your
`package.json` `typecheck` script, and does not run eslint. Both are arbitrary
shell/JS from the repository, and this check is invoked unattended with no
approval — running them would hand a hostile repo code execution the moment the
agent verified.

To be precise about what `tsc --noEmit` does: it bypasses your package scripts
and writes no build output (no JS, no `.d.ts`, no `.tsbuildinfo`), but it is
still a full type-check. It reads `tsconfig.json`, every source file that
config includes, everything those files import, the `.d.ts` declarations they
pull in, and the `package.json` metadata needed to resolve modules. "Reads only
config" would be wrong; "emits nothing" is the accurate part.

The other commands do not modify your source, but none of this is
side-effect-free on disk: `python -m compileall` writes `__pycache__/`, and
`cargo check` writes to `target/`.

While the check is red, the model **cannot** end the turn: the diagnostics are
injected and another round is forced, up to a bounded retry count so an
unfixable project can't spin forever. A check that times out or fails to run is
recorded as *unverified*, never as clean — and running out of retries does not
promote unverified to verified.

Note the gate runs a typecheck, not your tests.

### 3. Syntax checks before anything else

Writes and edits are syntax-checked in-process with tree-sitter as they happen,
so an obviously broken edit is caught in the same round rather than costing a
shell run.

### 4. Open todos push back on a bare "done"

If the model wrote a todo list during the turn and items are still open, a
closing message is rejected and another round is forced, naming the next item.

This one is bounded, not absolute: there is a cap on continuations, plus a
diminishing-returns guard that gives up after three rounds in which the open
count did not fall. A model that cannot advance its own plan is allowed to
stop rather than being nagged forever — so open todos are not a permanent
block.

## What is not enforced

- **Your test suite.** localcode never runs `pytest` or `npm test` on its own
  initiative. The model decides to run tests; the enforcement above is about
  whether a passing run was *recorded*, not about causing one.
- **Anything outside `build_app` / `edit_existing`.** A request the classifier
  reads as `general_task`, `run_or_launch` or `question` carries no evidence
  guarantee, even if it changed code. So does a build/edit turn that changed
  no code files.
- **Correctness.** A green checker means the project compiles and the commands
  the model chose passed. It is not a review.

So: not every turn ends on passing evidence. A turn localcode read as a build
or an edit, in which code files changed, either carries that evidence or is
reported as incomplete. Other turns end when the model stops calling tools.

## You still read the diff

```sh
git diff
```

[`/undo`](/localcode/guides/undo) reverts the agent's file changes if the
answer is no.
