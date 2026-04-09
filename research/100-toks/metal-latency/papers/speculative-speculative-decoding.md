# Speculative Speculative Decoding (Saguaro)

**Paper**: https://arxiv.org/abs/2603.03251
**Authors**: Tanishq Kumar, Tri Dao, Avner May
**Date**: March 2026

## Core Idea

Standard speculative decoding has a sequential dependency: draft -> verify -> draft -> verify.
Saguaro parallelizes this by pre-speculatively drafting during verification.

While verification N is running on the target model, the draft model predicts likely
verification outcomes and prepares speculations for them. If the actual outcome is in
the predicted set, the next speculation is returned immediately -- zero drafting latency.

## Architecture

- Multiple speculation "slots" run in parallel with verification
- Cache of pre-computed speculations indexed by verification outcome
- Fallback to standard speculative decoding on cache miss
- Adaptive: at low batch sizes, uses full draft model; at high batch sizes, switches
  to low-latency draft model

## Performance

- ~30% faster than strongest speculative decoding baselines
- Up to 5x faster than autoregressive generation
- Improves throughput-latency Pareto frontier across batch sizes

## Relevance to Our Problem

**Limited relevance for single-user local inference.**

Saguaro is designed for multi-GPU server scenarios where you can dedicate separate
devices to speculation vs verification. On a single Apple M4 with one GPU, there's
no way to run parallel draft+verify without time-sharing the same GPU.

However, the **pipelining insight** is transferable: if we could overlap command buffer
N+1 encoding with command buffer N execution, we eliminate the CPU-GPU gap. This is
exactly the MTLSharedEvent double-buffering approach (see solutions.md).

The key takeaway: **the overhead between decode steps matters enormously**, and even
partial overlap of work across the CPU-GPU boundary can yield 30%+ gains.
