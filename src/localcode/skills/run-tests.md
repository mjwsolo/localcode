---
name: run-tests
description: Run the project's AUTOMATED TEST SUITE (pytest / jest / cargo test / go test) to verify correctness. DO NOT use this to launch or start a user-facing app — for running an app, execute its entry point directly via the bash tool.
when_to_use: User explicitly says "run tests", "run the tests", "are tests passing", "check the tests". NOT when they say "run it" or "run the app" — that means launch the app, not the test suite.
---

Detect and run the test suite.

Framework detection — check in this order:
1. `pyproject.toml` has `[tool.pytest]` or `pytest` in dependencies → `pytest -q`
2. `package.json` has a `test` script → `npm test` (or `pnpm`/`yarn` based on lockfile)
3. `Cargo.toml` exists → `cargo test`
4. `go.mod` exists → `go test ./...`
5. Fallback: look for `tests/`, `test/`, `__tests__/`. Ask the user if unclear.

Scope to affected files when possible (faster):
- `pytest tests/test_foo.py` for a change to `src/foo.py`
- `pytest -k <function-name>` for a change to a specific function
- `cargo test <mod>::` for a single Rust module

Report faithfully per rule 17:
- **Passing**: "N tests ran, all passed."
- **Failing**: quote the exact failure output (first 20 lines). Don't hide.
- **Crashed** (didn't run): "Tests crashed during collection: <error>. I did NOT run them."
