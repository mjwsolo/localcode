# Introduction

LocalCode is an AI coding assistant that runs entirely on your machine.

It uses **Gemma 4 26B** — a 25 billion parameter model from Google — running on Apple Silicon at 27 tokens per second with 32K context. No cloud, no API keys, no data leaving your laptop.

## Why local?

Cloud coding assistants are great, but they have tradeoffs:

- Your code gets sent to someone else's servers
- You need an internet connection
- You pay per token or per month
- Rate limits hit when you need it most

LocalCode removes all of that. The model runs on your hardware, the code stays on your disk, and it works offline.

## How it works

Under the hood, LocalCode runs a custom fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with **TurboQuant KV cache compression** — a technique from [Google's ICLR 2026 paper](https://arxiv.org/abs/2501.10208) that we patched into llama.cpp for Apple Silicon.

This is what makes the numbers work:

| What | How |
|------|-----|
| 26B model on 16GB | IQ3_S quantization (10.4GB) + mmap |
| 32K context on 16GB | TurboQuant compresses KV cache 3.8x → 355 MiB |
| 27 tok/s | Full Metal GPU offload via sysctl unlock |
| Native tool calling | Gemma 4's built-in tool tokens, parsed by llama-server |

## What can it do?

LocalCode is a coding agent. You tell it what you want, and it figures out how to do it:

- **Read and understand** your codebase
- **Write and edit** files with surgical precision
- **Run commands** — tests, builds, git operations
- **Search** your code by pattern or content
- **Reason through** complex multi-step problems (in thinking mode)

The model decides which tools to use. You don't need to tell it how to do its job.

## What it's not

- It's not a cloud service — nothing leaves your machine
- It's not a VS Code extension (yet) — it's a terminal tool
- It's not trying to replace Claude or GPT — it's the tool you use when you can't or don't want to send your code to the cloud

## Next steps

- [Getting Started](getting-started.md) — install and run
- [First Run](first-run.md) — what to expect
