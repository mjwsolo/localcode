# Configuration

All settings live in `~/.gem/config.toml`.

!!! note
    The config path will move to `~/.localcode/` in a future release.

## Config file

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

## Runtime modes

| Mode | Speed | Context | Thinking | GPU Required |
|------|-------|---------|----------|-------------|
| `turbo` | 27 tok/s | 32K | Off | Yes |
| `turbo-think` | 26 tok/s | 32K | On | Yes |
| `speed` | 18 tok/s | 10K | Off | No |
| `speed-think` | 17 tok/s | 10K | On | No |

!!! tip
    `speed` modes run on CPU only — no sysctl unlock needed. Useful for quick tasks or if you can't run sudo.

## Server flags

The server is launched automatically with these flags:

```bash
llama-server \
  --model <gguf-path> \
  --port 8081 \
  -ngl 999              # Full GPU offload
  --mmap                # Memory-map model from SSD
  -ctk q8_0             # Key cache: 8-bit quantization
  -ctv turbo4           # Value cache: TurboQuant 4-bit
  -fa on                # Flash attention
  -c 32768              # 32K context window
  --threads 10          # CPU threads for expert computation
  -b 2048 -ub 512       # Batch sizes
  -np 1                 # Single slot
  -fit off              # Bypass auto-fitter
  --cache-ram 0         # Disable cross-request prompt cache
```

## GPU memory unlock

Required for turbo modes on 16GB Macs:

```bash
sudo sysctl iogpu.wired_limit_mb=14336
```

- Raises Metal GPU working set from ~11GB to 14GB
- Resets on reboot (safe, no permanent changes)
- Auto-prompted on launch
- Not needed on 24GB+ Macs or for CPU-only modes

## Environment variables

| Variable | Description |
|----------|-------------|
| `GEM_KV_CACHE_TYPE_K` | Override K cache type |
| `GEM_KV_CACHE_TYPE_V` | Override V cache type |
| `GEM_TEMPERATURE` | Override temperature |
| `GEM_MAX_CONTEXT_CHARS` | Override context budget |
