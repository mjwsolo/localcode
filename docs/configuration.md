# Configuration

## Config File

Located at `~/.gem/config.toml`. Created on first run.

```toml
[runtime]
provider = "llama_cpp"
base_url = "http://localhost:8081"
model = "gemma26b-iq3"
laptop_26b_runtime_mode = "turbo"

# KV cache (TurboQuant)
kv_cache_type_k = "q8_0"
kv_cache_type_v = "turbo4"

# Server binary
llama_cpp_binary = "/path/to/llama-cpp-turboquant/build/bin/llama-server"

# Generation
temperature = 0.7
max_context_chars = 40000
```

## Runtime Modes

| Mode | Speed | Context | Thinking |
|------|-------|---------|----------|
| `turbo` | 27 tok/s | 32K | Off |
| `turbo-think` | 26 tok/s | 32K | On |
| `speed` | 18 tok/s | 10K | Off (CPU only, no sysctl needed) |
| `speed-think` | 17 tok/s | 10K | On (CPU only) |

## Server Launch Flags

The server is launched automatically. These are the flags used:

```bash
llama-server \
  --model <gguf-path> \
  --port 8081 \
  -ngl 999              # Full GPU offload
  --mmap                # Memory-map model from SSD
  -ctk q8_0             # Key cache: 8-bit quantization
  -ctv turbo4           # Value cache: TurboQuant 4-bit (3.8x compression)
  -fa on                # Flash attention
  -c 32768              # 32K context window
  --threads 10          # CPU threads for expert computation
  -b 2048 -ub 512       # Batch sizes
  -np 1                 # Single slot
  -fit off              # Bypass auto-fitter (we manage memory)
  --cache-ram 0         # Disable cross-request prompt cache
```

## GPU Memory Unlock

Required for turbo modes. Raises Metal GPU working set from ~11GB to 14GB.

```bash
sudo sysctl iogpu.wired_limit_mb=14336
```

- Resets on reboot (safe)
- App auto-prompts on launch
- Not needed for CPU-only modes (speed/speed-think)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEM_KV_CACHE_TYPE_K` | Override K cache type |
| `GEM_KV_CACHE_TYPE_V` | Override V cache type |
| `GEM_TEMPERATURE` | Override temperature |
| `GEM_MAX_CONTEXT_CHARS` | Override context budget |
| `TURBO_USE_WHT` | Set to 1 to enable WHT rotation (default: off for tool compatibility) |
