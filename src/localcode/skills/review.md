---
name: review
description: Review code changes for correctness, security, performance, style — specific line-numbered feedback.
when_to_use: User asks for a review. Also as a second-pass sanity check after a non-trivial diff. Prefer this over informal "does this look right?"
---

Review the diff (or scope the user named).

Focus in priority order:
1. **Correctness** — bugs, off-by-one, wrong conditions, unhandled edge cases.
2. **Security** — injection (SQL/command/XSS), auth gaps, data exposure, unsanitized input crossing a trust boundary.
3. **Performance** — obvious N+1 queries, unbounded loops, blocking I/O on hot paths.
4. **Style** — matches project conventions? Naming consistent?

Process:
- `git diff` (or the user-named scope) first. Don't review what you haven't seen.
- For each issue, cite `path:line` and explain why it matters.
- Suggest fixes as `edit_file` calls when confident. Describe and defer when unsure.
- Don't flag stylistic preferences that aren't in the project's style. Don't nitpick.
- End with 1-line verdict: ship / fix-first / blocked.

If the diff is clean, say so briefly. A 2-line "LGTM, one tiny suggestion" review beats a 40-line manufactured-concerns one.
