---
name: locate
description: Find the exact file → class/function → line range to edit BEFORE reading or changing anything. Prevents aimless exploration.
when_to_use: User asks to change/fix/review a specific behavior without naming the file, or the task is "where does X happen?". First step for any non-trivial edit on an unfamiliar repo.
---

Locate the target before acting. Based on the Agentless pattern
(Xia et al. 2024) — the strongest small-model SWE-bench delta came
from structured localization, not better planning.

Process:
1. **Surface search** — `grep -rn "<the name-ish thing from the request>" src/ --include='*.py'`. Start wide, don't assume the exact symbol spelling.
2. **Narrow to a file** — pick the most-referenced or most-obvious match. If 3+ files look equally relevant, say so and ask the user which one.
3. **Inside that file, find the region** — `grep -En 'def <name>|class <name>' <file>` to get the line number of the specific function/class. (`-E` uses extended regex so `|` alternates without backslash.)
4. **Read a bounded window** — `read_file(path=<file>, offset=<start-10>, limit=80)`. Don't read the whole file; 2-3 bit models choke on 1500-line blobs.
5. **State what you found BEFORE editing**: "Target: `path/to/file.py:123` — `def foo(...)`. Making change because <reason>."

Anti-patterns (don't do):
- `list_files(.)` as first move when the user named a specific behavior.
- Reading a full file when a grep would pinpoint 40 lines.
- Editing before stating what you found.
