---
title: Models & Performance
description: The model catalogue, and how to think about speed on your Mac.
---


## What determines speed

Three things matter, roughly in this order:

1. **Active parameters per token.** On Apple Silicon, memory bandwidth limits decoding speed. A Mixture-of-Experts model uses only a few billion parameters per token. It reads far fewer bytes per token than a dense model with the same total size.
2. **Memory bandwidth.** This varies much more between chip tiers than the number of cores. It directly affects decoding speed.
3. **KV cache size.** TurboQuant compression (`q8_0`-K + `turbo4`-V) keeps the cache small enough for long contexts to remain practical. See [Unified Memory](/localcode/concepts/unified-memory).

## The tok/s numbers in the model picker

The model picker shows an estimated decoding speed next to each quantisation. This number is **calculated, not measured**. localcode does not run a benchmark on your machine. There is no benchmark command or benchmark screen.

The estimate uses an analytic model. It divides the bytes read per token by an assumed share of your chip's rated memory bandwidth. It then adds a fixed compute time for each token. The bytes per token come from the quant's size and its active-parameter fraction. This means MoE models count only their active experts. The model is calibrated using a small number of maintainer measurements from one machine.

Use the estimate to compare options. It can show that one quant will be slower than another on your hardware. Do not treat it as a prediction of your actual throughput. Real speed also depends on context length, thermal state, and other running tasks.

## Checking your setup

```text
/status     # server health, current model, perf configuration
```
