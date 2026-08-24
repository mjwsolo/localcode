---
title: Error Codes
description: Stable Eccc codes for every user-facing error.
---

Every error shown to localcode users has a stable `Eccc` code. This makes it easy to search for and refer to an error across versions.

| Range | Area |
| --- | --- |
| `E1xxx` | Setup and startup - starting the server, model files, and memory |
| `E2xxx` | Tool handling - unknown tools, invalid arguments, and permission or hook denials |
| `E3xxx` | Runtime and model |

The full table is **generated from the code**. It is not written by hand. The source of truth is `src/localcode/errors.py`. The generated table is in [`docs/ERRORS.md`](https://github.com/mjwsolo/localcode/blob/main/docs/ERRORS.md) in the repository. Regenerate it with:

```sh
python -m localcode.errors --emit-docs > docs/ERRORS.md
```

Detailed technical information about the latest project error is written to `<project>/.localcode/last_error.log`.

:::note[There is no `localcode setup` command]
Setup runs inside the TUI on first launch. An older generated copy of the error table still tells users to run `localcode setup` for `E1001`, `E1002`, and `E1003`. Ignore that: relaunch localcode and let the TUI handle setup and the model download.
:::
