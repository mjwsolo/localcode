# Architecture

## Overview

```
User Input
    │
    ▼
┌──────────────────┐
│   CLI (cli.py)   │  Mode picker, auto-sysctl, auto-server-start
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   App (app.py)   │  Session management, context building, message composition
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Agent Loop      │  Intent classification → executor routing
│  (agent_loop.py) │  CREATE | EDIT | CHAT | RUN | SEARCH | FIX | REVIEW
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Runtime         │  HTTP client to llama-server
│  (runtime.py)    │  Streaming, tool call parsing, thinking extraction
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  TurboQuant llama-server (C++)           │
│  ┌─────────────┐  ┌──────────────────┐   │
│  │ Gemma 4 26B │  │ TurboQuant KV    │   │
│  │ IQ3_S 10.4GB│  │ turbo4 (3.8x)    │   │
│  └─────────────┘  └──────────────────┘   │
│  ┌─────────────┐  ┌──────────────────┐   │
│  │ Metal GPU   │  │ Tool Call Parser │   │
│  │ ngl=999     │  │ Gemma 4 native   │   │
│  └─────────────┘  └──────────────────┘   │
└──────────────────────────────────────────┘
```

## Model

**Gemma 4 26B-A4B** — Google's Mixture-of-Experts model:
- 25.2B total parameters, **3.8B active** per token
- 128 experts, top-8 routing + 1 shared expert
- IQ3_S quantization (3.5 bits/weight, 10.4GB)
- 256K native context window

The MoE architecture is key — only 3.8B params are computed per token, giving the speed of a 4B model with the knowledge of a 25B model.

## Inference Server

We use a patched `llama.cpp` with TurboQuant KV cache compression:

- **KV Cache**: `q8_0` keys + `turbo4` values (3.8x compression)
- **32K context** in 355 MiB (vs ~1.6GB without TurboQuant)
- **Full Metal GPU offload** via `sysctl iogpu.wired_limit_mb=14336`
- **2 graph splits** (optimized MoE dispatch)
- **Native Gemma 4 tool calling** (upstream parser cherry-picked)

## Message Flow

1. User types a message
2. `classify_intent()` determines task type (CREATE, EDIT, CHAT, etc.)
3. System prompt selected based on intent
4. Context gathered (project files, git status) for substantive tasks
5. Messages sent to llama-server via `/v1/chat/completions`
6. Response parsed for tool calls or content
7. Tool results fed back for multi-turn loops
8. Final response displayed

## File Structure

```
localcode/
├── src/gem/
│   ├── cli.py              # Entry point, mode picker
│   ├── app.py              # App lifecycle, session management
│   ├── agent_loop.py       # Intent routing, executors
│   ├── runtime.py          # llama-server HTTP client
│   ├── composer.py         # Message composition
│   ├── context_manager.py  # System prompts, context building
│   ├── toolkit.py          # Tool definitions (read_file, bash, etc.)
│   ├── output.py           # Terminal display
│   └── config.py           # Configuration
├── llama-cpp-turboquant/   # Patched llama.cpp fork
│   ├── ggml/               # TurboQuant ggml layer
│   ├── src/                # llama.cpp core
│   ├── common/             # Chat parser (upstream fixes)
│   └── BUILD.sh            # One-command build
└── docs/                   # Documentation
```
