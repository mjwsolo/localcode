---
title: Error Codes
description: Stable Eccc codes for every user-facing error.
---

Every error shown to localcode users has a stable `Eccc` code. This makes it easy to search for and refer to an error across versions.

| Range | Area |
| --- | --- |
| `E1xxx` | Setup and startup — starting the server, model files, and memory |
| `E2xxx` | Tool handling — unknown tools, invalid arguments, and permission or hook denials |
| `E3xxx` | Runtime and model |

The full table is **generated from the code**. It is not written by hand. The source of truth is `src/localcode/errors.py`. The generated table is in [`docs/ERRORS.md`](https://github.com/mjwsolo/localcode/blob/main/docs/ERRORS.md) in the repository. Regenerate it with:

```sh
python -m localcode.errors --emit-docs > docs/ERRORS.md
```

Detailed technical information about the latest project error is written to `<project>/.localcode/last_error.log`.

:::caution[The checked-in table is out of date]
`src/localcode/errors.py` is correct. It has **no** reference to a `localcode setup` command. However, the committed `docs/ERRORS.md` was generated from an older registry. It still tells users to "Run `localcode setup`" for `E1001`, `E1002`, and `E1003`. Regenerating the file fixes this. The registry does not need any changes. Regeneration is outside the scope of this preview. This is why the link above may not match the current fix instructions.
:::
