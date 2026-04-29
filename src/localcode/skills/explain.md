---
name: explain
description: Walk through a file or function for a newcomer — purpose, flow, gotchas, concise.
when_to_use: User asks "explain X" / "how does Y work" / "I don't understand this". Also as a pre-edit orientation step before a non-trivial change.
---

Explain the target code.

Structure:
1. **Purpose** — one sentence. What problem does this solve?
2. **Flow** — walk through the key logic in order of execution. Skip boilerplate (imports, trivial setters).
3. **Gotchas** — 1-3 things a reader might miss:
   - Non-obvious state mutations
   - Side effects (I/O, globals, env vars)
   - Subtle ordering dependencies
   - Comments or naming that's misleading
4. **What it does NOT do** — common misconceptions.

Concise. Reference `path:line` for key chunks. End with "Ask follow-ups if anything is unclear" — don't re-summarize. Small models tend to repeat themselves; resist.
