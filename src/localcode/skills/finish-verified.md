---
name: finish-verified
description: Finish a coding task once each requested outcome has concrete evidence.
when_to_use: Before declaring a coding task complete or when work is drifting into optional polish.
---

1. Restate the requested outcomes as a short checklist.
2. Attach one concrete proof to each outcome: a focused test, build, runtime probe, or exact file inspection.
3. Treat exit code zero as shell success, not automatically task success. Check output for masked errors, fallbacks, skipped checks, and pipelines that hide an upstream failure.
4. Resolve only missing requested outcomes. Do not add unrelated cleanup or polish.
5. When every outcome has evidence, stop and report the evidence. If one cannot be proven, name that gap precisely.
