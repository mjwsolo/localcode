# Gem

Gem is a local-first AI coding assistant for developers who want an open-source, terminal-native workflow with Gemma 4 running on their own machine.

This repo currently contains two things:

- `src/gem/`: the new Python implementation for a pragmatic v1.
- `gem_code/`: the legacy TypeScript/Ink codebase snapshot that was inspected during the migration.

## Why The Refactor

The legacy app is a large TypeScript terminal product with deep coupling to hosted APIs, remote control, analytics, MCP plumbing, and product-specific services. The parts worth preserving conceptually are:

- interactive terminal workflow
- streaming assistant output
- background-safe shell/task execution
- session persistence
- repo-aware context gathering
- diff-oriented coding flow

The new Python version keeps those ideas and drops the product-specific sprawl.

## Interface Choice

A TUI is the right v1 interface for Gem because the target user is already living in a terminal while coding. It keeps file paths, git, diffs, shell output, and model interaction in one place without introducing browser or hosted-service requirements.

Practical implication: this version is intentionally terminal-first, keyboard-driven, and local. It does not attempt to recreate every feature from the legacy app.

## Runtime Choice

Gem targets **Ollama** first for local Gemma 4 because it is the simplest practical setup:

- one local HTTP endpoint
- streaming responses
- simple install story
- no hosted dependency

Gem now also has:

- an experimental `llama_cpp` provider path for local server setups that expose an OpenAI-style `/v1/chat/completions` endpoint. This is the preferred direction for aggressive low-latency tuning on constrained hardware.
- an `mlx-local` provider path for Apple Silicon users running MLX quantized Gemma models locally.
- an advanced `huggingface-local` provider path for users who want to run Gemma checkpoints directly through local `transformers` + `torch`.
- a local browser preset through Playwright MCP.
- a local voice stack with `whisper.cpp` or `faster-whisper` for STT and `kokoro` or `piper` for TTS.

Recommended stack:

- default onboarding: `ollama`
- Mac performance path: `mlx-local`
- cross-platform performance path: `llama_cpp`
- advanced custom local backend: `huggingface-local`

## Gemma 4 Design

Gem is now explicitly centered on Gemma 4.

Official Gemma docs currently state:

- Gemma 4 released on **March 31, 2026**
- Gemma 4 ships in **E2B, E4B, 31B, and 26B A4B / MoE-style** variants
- Gemma 4 supports **text, image, and audio input**
- Gemma 4 supports context windows up to **256K**

Practical implications for Gem:

- setup starts with Gemma 4 profile selection
- prompts and context budgets vary by selected Gemma 4 tier
- tool use is exposed as a local-first coding workflow, not as cloud orchestration
- the assistant stays useful on both small and large local hardware

## Installation and Running

To install dependencies and run the example script, follow these steps:

### 1. Install Dependencies

Navigate to the root directory of the project and install the required Python packages using `pip`:
