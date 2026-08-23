---
title: Error Codes
description: What the E-codes in localcode's error messages mean.
---

Every error localcode shows carries a code like `E1010`, so you can search for it and refer to it across versions.

| Range | Area |
| --- | --- |
| `E1xxx` | Setup and startup: the server, model files, memory |
| `E2xxx` | Tools: unknown tool, bad arguments, a blocked or denied call |
| `E3xxx` | Runtime: the model server or the model itself |

The full table, with a fix for each code, is in [`docs/ERRORS.md`](https://github.com/mjwsolo/localcode/blob/main/docs/ERRORS.md).

Details of the most recent error are written to `<project>/.localcode/last_error.log`.
