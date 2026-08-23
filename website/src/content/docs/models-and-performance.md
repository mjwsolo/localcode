---
title: Models & Performance
description: Every model in the picker, and what decides speed on your Mac.
---

## The models

localcode ships a short list of models rather than a model zoo. Every one runs on the binaries included in the package; the only download is the weights. localcode recommends one by your Mac's memory and marks it with a star. You choose.

| Model | Weights | Quant | Active params | Min RAM |
| --- | ---: | --- | --- | ---: |
| Gemma 4 12B | 7.4 GB | UD-Q4_K_XL | 12B (dense) | 16 GB |
| Qwen 3.6 35B-A3B | 10.7 GB | UD-IQ2_M | 3.0B (MoE) | 24 GB |
| Gemma 4 26B-A4B | 11.2 GB | UD-IQ3_S | 3.8B (MoE) | 24 GB |
| DiffusionGemma 26B-A4B | 15.7 GB | Q4_K_M | 4B (diffusion MoE) | 32 GB |
| Muse Glimmer 30B | 15.9 GB | UD-Q4_K_XL | 30B (dense, vision) | 32 GB |
| Qwen 3.8 27B | 17.9 GB | UD-Q4_K_XL | 27B (dense) | 36 GB |
| North-Mini-Code 30B-A3B | 17.9 GB | UD-Q4_K_M | 3B (MoE) | 36 GB |
| Gemma 4 12B (full) | 23.8 GB | BF16 | 12B (dense) | 48 GB |
| Gemma 4 26B-A4B | 28.0 GB | UD-Q8_K_XL | 3.8B (MoE) | 64 GB |
| Qwen 3.6 35B-A3B | 38.5 GB | UD-Q8_K_XL | 3.0B (MoE) | 96 GB |

Min RAM is the memory at which localcode will recommend the model. You can pick a heavier one by hand. DiffusionGemma is a research model that is never recommended automatically; it runs through a separate bundled binary.

The picker also lets you browse other quantisations of each model. See [Choose a Model](/localcode/start-here/choose-a-model).

## What decides speed

1. **Active parameters per token.** Generation on Apple Silicon is limited by memory bandwidth, so a Mixture-of-Experts model that reads only a few billion parameters per token is much faster than a dense model of the same size.
2. **Your chip's memory bandwidth.** This differs more between chip tiers (M4 vs M4 Pro vs M4 Max) than the core count does.
3. **Context length.** A longer conversation means a bigger KV cache and slower turns. See [Unified Memory](/localcode/concepts/unified-memory).

## The tok/s figure in the picker

The speed next to each quant is an estimate for your chip, not a measurement. It is good for comparing options, for example to see that one quant will be noticeably slower than another. Real speed also depends on context length, heat and what else is running.

One measured reference: a MacBook Pro with an M5 Max and 128 GB, running Qwen 3.6 35B-A3B UD-IQ2_M at a 131072-token context, generates about 89 tokens/s and processes prompts at about 1174 tokens/s.

## Checking your setup

```text
/status     # server health, current model, performance settings
```
