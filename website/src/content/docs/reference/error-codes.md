---
title: Error Codes
description: Stable Eccc codes for every user-facing error.
---

Every user-facing error in localcode carries a stable `Eccc` code, so an error
can be searched for and referenced across versions.

| Range | Area |
| --- | --- |
| `E1xxx` | Setup / startup — server launch, model files, memory |
| `E2xxx` | Tool dispatch — unknown tool, malformed arguments, permission or hook denial |
| `E3xxx` | Runtime / model |

The full table is **generated from the code**, not hand-written. The
authoritative registry is `src/localcode/errors.py`, and the rendered table
lives in [`docs/ERRORS.md`](https://github.com/mjwsolo/localcode/blob/main/docs/ERRORS.md)
in the repository. Regenerate it with:

```sh
python -m localcode.errors --emit-docs > docs/ERRORS.md
```

Verbose technical detail for the most recent error in a project is written to
`<project>/.localcode/last_error.log`.

:::note[Preview stub]
This preview site links to the generated table rather than duplicating it. A
future pass should import `docs/ERRORS.md` into this site at build time so the
codes are searchable here, and correct the two remediation strings in the
registry that still reference a `localcode setup` command that does not exist.
:::
