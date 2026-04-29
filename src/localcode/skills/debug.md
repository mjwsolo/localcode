---
name: debug
description: Diagnose a bug — reproduce, read, hypothesize, fix, verify. Don't guess.
when_to_use: User reports an error, a test is failing, unexpected output, or "this doesn't work".
---

Debug the reported issue. Follow in order, don't skip.

1. **Reproduce.** Run the exact command or code path that surfaced the bug. Capture the actual error text — don't paraphrase it.
2. **Read the relevant code.** Use `grep` / `read_file` to find the failing location. Read enough surrounding context to understand what was expected. (Use the `locate` skill if the target isn't obvious.)
3. **Form ONE hypothesis** about the root cause. Write it in your reply before acting. "I think X because Y."
4. **Make the minimal fix.** If you're wrong, the error will still be there — that's fine, iterate. (Rule 16: diagnose before retrying.)
5. **Verify.** Re-run the reproduction command. If it now passes, use `edit-verified` / `run-tests` to confirm no regressions.

Don't:
- Guess fixes before reading the code.
- "Fix" by disabling the failing assertion / test.
- Add try/except that swallows the error.
- Declare victory without re-running the repro.

If you can't reproduce → ask for the exact command, environment, and full error output. Guessing wastes the user's time.
