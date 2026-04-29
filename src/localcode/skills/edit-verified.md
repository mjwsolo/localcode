---
name: edit-verified
description: Make a code edit, then IMMEDIATELY verify it with syntax check + targeted test run before declaring done.
when_to_use: Any time you call write_file or edit_file. This is the SWE-agent linter-gate pattern — the ablation evidence for small-model edit reliability.
---

Edit + verify in one atomic flow. Based on the SWE-agent linter-gate
ablation (Yang et al. 2024) — guarded edits were the single largest ACI
delta for small models.

After every `edit_file` or `write_file`, run in this order:

1. **Syntax check** (language-appropriate, non-blocking on stdout):
   - Python: `python -c "import ast; ast.parse(open('<path>').read())"`
   - TS/JS: `node --check <path>` (JS only) or `npx tsc --noEmit <path>`
   - Go: `gofmt -e <path>` (errors if syntax bad)
   - Rust: covered by `cargo check` in step 2

2. **Targeted test run** — only tests that could plausibly exercise this change:
   - Python: `pytest tests/test_<same-basename>.py -q` if it exists; else scope by keyword `pytest -k <function-name>`
   - JS: `npm test -- <path-to-test>` or `vitest run <path>`
   - Rust: `cargo test <test-name>`
   - Go: `go test ./<pkg> -run <TestName>`

3. **If step 1 or 2 fails**: report the error verbatim, fix, re-run. Do NOT claim done.
   If step 1 passes and step 2 has no matching test: say "syntax OK, no test covers this change" — don't imply tests ran when they didn't (rule 17).

4. **If no tests exist anywhere**: run the smallest reproducer that exercises the change (import the module, call the function with a trivial input). Report what you ran.

Don't: claim "should work", "this is correct", or "the fix is straightforward" without the verify step. Don't re-run identical failing commands (rule 16).
