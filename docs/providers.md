# Providers

LocalCode supports multiple inference backends. Not all are equally polished — here's the honest breakdown.

## Recommended path

### 1. TurboQuant llama.cpp (default)

The intended experience. Custom llama.cpp fork with TurboQuant KV cache compression.

- **Best for**: Maximum performance, 32K context, full tool calling
- **Speed**: 27 tok/s decode
- **Setup**: Automatic on first run (builds from source)

This is what the project is built around. Everything else is a fallback.

### 2. Ollama

Easiest way to get started if you already have Ollama installed.

- **Best for**: Quick setup, trying LocalCode without building anything
- **Speed**: ~28 tok/s (but limited to 4-8K context)
- **Setup**: `ollama pull gemma3:27b` then set provider in config

!!! warning "Context limitation"
    Ollama doesn't support TurboQuant. Without it, 32K context doesn't fit on 16GB — you'll be limited to ~4-8K tokens.

### 3. MLX

Apple's native ML framework. Fast at small context, but the 4-bit model is 15GB.

- **Best for**: Apple Silicon users who want MLX-native inference
- **Speed**: Fast at 1K context, needs benchmarking at larger sizes
- **Setup**: Requires 24GB+ RAM (15GB model doesn't fit on 16GB without swap)

### 4. HuggingFace Transformers

Direct Python inference. Slowest but most flexible.

- **Best for**: Advanced users, custom model experiments
- **Speed**: Slowest option
- **Setup**: `pip install transformers`

## Choosing a provider

The simplest way:

```bash
localcode setup
```

This detects your hardware and recommends the right provider and model. Follow its advice unless you have a specific reason not to.

## The honest take

The 26B Gemma model on TurboQuant llama.cpp is the intended LocalCode experience. Everything else is either a convenience fallback (Ollama) or experimental (MLX, HuggingFace). The docs don't pretend they're all equal.
