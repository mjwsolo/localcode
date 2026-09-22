---
title: Error Codes
description: Stable Eccc codes for every user-facing error.
---

Every error shown to localcode users has a stable `Eccc` code. This makes it easy to search for and refer to an error across versions.

| Range | Area |
| --- | --- |
| `E1xxx` | Setup and startup - starting the server, model files, and memory |
| `E2xxx` | Tool handling - unknown tools, invalid arguments, and permission denials |
| `E3xxx` | Runtime and model |

The full table is **generated from the code**. It is not written by hand. The source of truth is `src/localcode/errors.py`. The generated table is in [`docs/ERRORS.md`](https://github.com/mjwsolo/localcode/blob/main/docs/ERRORS.md) in the repository. Regenerate it with:

```sh
python -m localcode.errors --emit-docs > docs/ERRORS.md
```

When the model server fails to load a model, the detail is in `server.log` under the run directory (`~/.local/share/localcode-agent/run` by default). The classic interface and `localcode run` write `<project>/.localcode/last_error.log`.

:::note[`dyld: Library not loaded` on launch]
If `localcode` fails right away with a `dyld` error such as "Library not loaded", the Mac is running a macOS older than 13. The bundled binaries are built for macOS 13 and newer on Apple Silicon. Update macOS; there is no build for older versions.
:::

:::note[There is no `localcode setup` command]
Setup runs inside the interface on first launch. An older generated copy of the error table still tells users to run `localcode setup` for `E1001`, `E1002`, and `E1003`. Ignore that: relaunch localcode and choose a model in the picker.
:::
