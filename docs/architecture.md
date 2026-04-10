# Architecture

How LocalCode works under the hood.

## Overview

```mermaid
graph TD
    A[User Input] --> B[CLI · cli.py]
    B --> C[App · app.py]
    C --> D[Agent Loop · agent_loop.py]
    D --> E[Runtime · runtime.py]
    E --> F[TurboQuant llama-server]

    B -.- B1[Mode picker\nAuto-setup\nSysctl unlock]
    C -.- C1[Session management\nContext building\nMessage composition]
    D -.- D1[Tool calling\nMulti-turn execution\nIntent routing]
    E -.- E1[HTTP streaming\nTool call parsing\nThinking extraction]
    F -.- F1[Gemma 4 26B · Metal GPU\nTurboQuant KV · mmap]

    style A fill:#18E299,color:#000
    style F fill:#15803D,color:#fff
```

## Message flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as App
    participant L as Agent Loop
    participant R as Runtime
    participant S as llama-server

    U->>A: "refactor this function"
    A->>L: Build context + messages
    L->>R: POST /v1/chat/completions
    R->>S: Stream request
    S-->>R: Tool call: read_file
    R-->>L: Parse tool call
    L->>L: Execute read_file
    L->>R: Tool result → next request
    R->>S: Stream request
    S-->>R: Tool call: edit_file
    R-->>L: Parse tool call
    L->>L: Execute edit_file
    L->>R: Tool result → next request
    R->>S: Stream request
    S-->>R: "Done. Refactored X to Y."
    R-->>A: Final response
    A-->>U: Display result
```

## The model

**Gemma 4 26B-A4B** — Google's Mixture-of-Experts model:

```mermaid
graph LR
    T[Input Token] --> R[Router]
    R --> E1[Expert 1]
    R --> E2[Expert 2]
    R --> E8[Expert 8]
    R -.-> E128[... 128 total]
    E1 --> M[Merge]
    E2 --> M
    E8 --> M
    S[Shared Expert] --> M
    M --> O[Output Token]

    style T fill:#18E299,color:#000
    style O fill:#18E299,color:#000
    style R fill:#4ADE80,color:#000
```

- 25.2B total parameters, **3.8B active** per token
- 128 experts, top-8 routing + 1 shared expert
- IQ3_S quantization (3.5 bits/weight, 10.4GB)
- 256K native context window

Only 3.8B params compute per token — speed of a 4B model, knowledge of a 25B model.

## Memory layout (16GB Mac)

```mermaid
pie title 16GB Unified Memory
    "Model (mmap from SSD)" : 10.4
    "KV Cache (TurboQuant)" : 0.355
    "Active GPU (attention)" : 2.0
    "OS + Apps" : 3.245
```

## Inference server

A patched llama.cpp fork with:

- **TurboQuant KV cache**: `q8_0` keys + `turbo4` values → 3.8x compression
- **32K context** in 355 MiB (vs ~1.6GB without TurboQuant)
- **Full Metal GPU offload** via sysctl memory unlock
- **2 graph splits** (optimized MoE dispatch)
- **Native Gemma 4 tool parsing** (upstream cherry-picked)

## Tool loop

```mermaid
flowchart TD
    A[User message] --> B{Model response}
    B -->|Tool call| C[Execute tool]
    C --> D[Append result to context]
    D --> E{Round < 15?}
    E -->|Yes| B
    E -->|No| F[Return text response]
    B -->|Plain text| F
    F --> G[Display to user]

    style A fill:#18E299,color:#000
    style G fill:#18E299,color:#000
```

## File structure

```
localcode/
├── src/gem/
│   ├── cli.py              # Entry point, mode picker
│   ├── app.py              # Session management
│   ├── agent_loop.py       # Tool calling, execution
│   ├── runtime.py          # llama-server HTTP client
│   ├── composer.py         # Message composition
│   ├── context_manager.py  # System prompts, context
│   ├── toolkit.py          # Tool definitions
│   ├── output.py           # Terminal display
│   └── config.py           # Configuration
├── docs/                   # This documentation
└── tests/                  # Test suite
```

## Design decisions

| Decision | Why |
|----------|-----|
| Custom llama.cpp fork over Ollama | Ollama doesn't support TurboQuant — can't do 32K context on 16GB |
| Full GPU offload (`-ngl 999`) | 27 tok/s with 2 graph splits vs 18 tok/s CPU-only |
| Native tool calling over regex routing | Gemma 4 scores 85.5% on tool benchmarks — let the model decide |
| No hardcoded token budgets | Model generates until EOS (4096 safety net) |
| Thinking off for fast mode | IQ3_S thinking degenerates on simple prompts |

## What we tried and ruled out

| Approach | Result |
|----------|--------|
| MLX runtime | 4-bit model is 15GB — doesn't fit on 16GB without swap |
| Speculative decode | Slower for MoE (21.9 vs 27 tok/s) — active weights too small |
| Multi-token prediction | Gemma 4 not trained with MTP heads |
| Expert pruning (top-4) | Only 10% gain, quality risk |
| 128K context default | OOMs under memory pressure — 32K is stable |
