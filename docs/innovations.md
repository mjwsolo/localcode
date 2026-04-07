# Innovations

## 1. TurboQuant KV Cache on Apple Silicon

We patched Google's TurboQuant (ICLR 2026) into llama.cpp for Apple Silicon Metal. This compresses the KV cache 3.8x, enabling 32K context in 355 MiB on a 16GB MacBook.

**Without TurboQuant**: 32K context needs ~1.6GB KV cache → crashes on 16GB
**With TurboQuant**: 32K context needs 355 MiB → stable at 27 tok/s

### How it works
- Asymmetric compression: `q8_0` keys (preserve quality) + `turbo4` values (3.8x compression)
- PolarQuant: 4-bit quantization with 16 optimal centroids
- Walsh-Hadamard rotation: Gaussianizes values for better quantization
- Per-block norms: preserves vector magnitude

## 2. GPU Memory Unlock for 16GB Macs

Apple Silicon limits GPU memory to ~11GB working set by default. Our 10.4GB model + KV cache exceeds this. We solved it with:

```bash
sudo sysctl iogpu.wired_limit_mb=14336
```

This raises the Metal working set to 14GB. The app auto-prompts for this on launch. Resets on reboot (safe, no permanent changes).

**Result**: Full GPU offload with `ngl=999`, 2 graph splits, 27 tok/s decode.

## 3. Upstream Tool Calling Fix

The TurboQuant fork had outdated Gemma 4 chat parser files that broke native tool calling. We cherry-picked upstream fixes:

- `llama-vocab.cpp`: Gemma 4 BOS token fix (PR #21500)
- `chat-peg-parser.cpp`: Updated Gemma 4 tool call parsing
- `chat.cpp`: Updated template handling

**Result**: Native Gemma 4 tool calling (`<|tool_call>` tokens) works with TurboQuant at 24 tok/s.

## 4. Prompt Cache Management

TurboQuant's WHT rotation causes cross-request KV cache corruption when the server's prompt cache is enabled. We fixed this with `--cache-ram 0` which disables cross-request caching while keeping per-request KV cache (270ms TTFT after warmup).

## 5. MoE Mmap + GPU

On 16GB, the 10.4GB model can't fit entirely in GPU memory alongside KV cache. We use `--mmap` which memory-maps the model from SSD. macOS's unified memory architecture means the GPU reads directly from mmap'd pages — the MoE architecture only activates 3.8B of 25.2B params per token, so most expert weights stay cold on SSD.

## Performance Summary

| Metric | Value |
|--------|-------|
| Decode speed | 24-27 tok/s |
| TTFT (warm) | 220-270ms |
| Context window | 32K tokens |
| KV cache size | 355 MiB |
| Model size | 10.4GB (IQ3_S) |
| GPU memory | ~2GB active |
| Tool calling | ✅ Native Gemma 4 |
| Thinking mode | ✅ Working |

## Comparison

| | LocalCode | Ollama | Cloud API |
|--|-----------|--------|-----------|
| Speed | 24-27 tok/s | 28 tok/s | ~50 tok/s |
| Context | 32K | 4-8K | 128K+ |
| Tools | ✅ | ✅ | ✅ |
| Privacy | 100% local | 100% local | Cloud |
| Cost | $0 | $0 | $$/month |
| Offline | ✅ | ✅ | ❌ |
