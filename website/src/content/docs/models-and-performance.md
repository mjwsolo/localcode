---
title: Models & Performance
description: The model catalogue, and how to think about speed on your Mac.
---

## The catalogue

localcode ships a curated catalogue rather than a model zoo. Every entry
records its quantisation, download size, active parameter count, architecture,
and the memory threshold at which it becomes auto-recommendable.

Production-ready picks by memory size are in
[Choose a Model](/localcode/start-here/choose-a-model). The complete table,
including the experimental entries, is in the
[repository README](https://github.com/mjwsolo/localcode#models).

Experimental models are pickable but never auto-recommended: their
architectures need a runner other than the bundled server, which localcode
builds on first use. Treat them as experiments.

## What actually determines speed

Three things, in roughly this order:

1. **Active parameters per token.** Decode on Apple Silicon is
   memory-bandwidth-bound. A Mixture-of-Experts model activates only a few
   billion parameters per token, so it reads far fewer bytes per token than a
   dense model of the same total size.
2. **Memory bandwidth.** This varies across chip tiers far more than core
   count does, and it is the number that moves decode speed.
3. **KV cache size.** TurboQuant compression (`q8_0`-K + `turbo4`-V) keeps the
   cache small enough that long contexts remain workable — see
   [Unified Memory](/localcode/concepts/unified-memory).

## The tok/s numbers in the model picker

The model picker shows an estimated decode speed next to each quantisation.
That number is **calculated, not measured**. localcode does not run a benchmark
on your machine — there is no benchmark command and no benchmark screen.

The estimate is an analytic model: bytes read per token (the quant's size
scaled by its active-parameter fraction, so MoE models count only their active
experts) divided by an assumed fraction of your chip's rated memory bandwidth,
plus a fixed per-token compute floor. It is calibrated against a small number
of maintainer measurements on one machine.

Treat it as a ranking aid — it is good at telling you that one quant will be
slower than another on your hardware. Do not treat it as a prediction of the
throughput you will see, which also depends on context length, thermal state
and whatever else is running.

## Checking your own setup

```text
/status     # server health, current model, perf configuration
```

:::note[Preview stub]
This page deliberately quotes no throughput figures. The maintainers' measured
numbers exist but have only been validated on a narrow set of hardware, so
publishing them here as though they generalise would be misleading. A future
pass should add a proper methodology section and per-chip results.
:::
