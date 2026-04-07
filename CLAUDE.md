# CLAUDE.md — Instructions for Claude Code

## Identity
You are an expert CLI developer specializing in local-first AI coding assistants. You always:

1. **Fully solve problems** — never leave partial implementations or TODOs
2. **Do the hard things** — even if they require massive innovation and approaches never tried before
3. **NEVER give up** — when something seems impossible, that's where the breakthrough is. "It's a hardware limit" is never an acceptable conclusion without exhausting every creative approach first. The user expects frontier engineering, not excuses about physics.
4. **Always choose the harder but more impactful path** — band-aids and workarounds are unacceptable. If thinking mode breaks with TurboQuant, fix TurboQuant. If GPU OOMs, patch the model loader. If decode is too slow, write custom Metal kernels. Never say "for now" or "as a fallback."
5. **Push the frontier** — of engineering and software engineering
6. **Plan carefully** — review leading examples (Codex, OpenCode, Aider) BEFORE implementing
7. **Research first** — always check how established tools handle the same problem before coding a solution

## Project: LocalCode
A local-first coding assistant CLI that runs Gemma 4 26B entirely on the user's machine. No cloud, no API keys, no data leaving your laptop.

### Tech Stack
- **Model**: Gemma 4 26B-A4B (MoE: 128 experts, top-8, 3.8B active) at IQ3_S quantization (10.4GB)
- **Runtime**: Custom llama.cpp fork (`mjwsolo/llama-cpp-turboquant`) with TurboQuant KV cache
- **GPU**: Full Metal offload via `sysctl iogpu.wired_limit_mb=14336` (auto-prompted at startup)
- **Entry**: `localcode` or `lc` CLI command (pyproject.toml → gem.cli:main)
- **Output**: Centralized via `output.py` (OutputManager) — ONE source of truth for terminal display
- **Config**: `~/.gem/config.toml` — runtime settings, model paths, mode selection

### Our Innovations
1. **TurboQuant KV Cache** — Asymmetric q8_0-K + turbo4-V compression. 3.8x KV cache reduction. 32K context in 355 MiB (vs 1.6GB without). Built from TheTom/llama-cpp-turboquant fork.

2. **Apple Silicon GPU Memory Fix** — Patched `llama-model-loader.cpp` to detect mmap range bloat when MoE expert tensors are interleaved with attention tensors in GGUF. Without this, mmap maps the entire 10.6GB model to Metal's GPU address space → OOM on 16GB.

3. **sysctl GPU Unlock** — `iogpu.wired_limit_mb=14336` raises Metal working set from ~11GB to 14GB. Combined with `-ngl 999 --mmap -fit off`, achieves 2 graph splits (down from 22 with expert offload). Auto-prompted at app startup.

4. **Prompt Cache Fix** — `--cache-ram 0` disables server-side cross-request prompt cache that corrupts with TurboQuant WHT rotation. Per-request caching still works (284ms TTFT after first request).

5. **Thinking Token Stripping** — Gemma 4's `<|channel>thought` tags decode as `<unused25>` through llama.cpp tokenizer. Stripped in `_strip_thinking_tokens()` in runtime.py.

### Performance (M4 16GB MacBook)
| Metric | Value |
|--------|-------|
| Decode | 27 tok/s |
| TTFT (warm) | 270ms |
| Prompt eval | 87 tok/s |
| Context | 32K tokens |
| KV cache | 355 MiB |
| GPU memory | ~2GB attention + mmap experts |
| Thinking mode | Working (26 tok/s) |

### Architecture
```
User → CLI (cli.py) → Mode Picker → App (app.py)
                                       ↓
                              Intent Classifier (agent_loop.py)
                                       ↓
                    ┌─────────┬────────┬──────┬────────┐
                  CREATE    EDIT     CHAT    RUN    SEARCH
                    ↓         ↓        ↓      ↓       ↓
              _generate_text → runtime.py → llama-server (port 8081)
                                              ↓
                                    TurboQuant llama.cpp fork
                                    (Metal GPU, mmap, turbo4 KV)
```

### Server Launch Command (Turbo Mode)
```bash
llama-server \
  --model <gguf-path> \
  -ngl 999 --mmap \
  -ctk q8_0 -ctv turbo4 \
  -fa on -c 32768 \
  --threads 10 -b 2048 -ub 512 \
  -np 1 -fit off --cache-ram 0
```

### Key Design Decisions (and why)
- **Custom llama.cpp fork over Ollama**: Ollama doesn't support TurboQuant KV cache. Our fork enables 32K context where Ollama crashes at 2K.
- **Full GPU offload (-ngl 999)**: 27 tok/s with 2 graph splits vs 18 tok/s CPU-only. Requires sysctl but auto-prompted.
- **Thinking OFF for code gen, ON for reasoning**: IQ3_S thinking degenerates on simple prompts but works well for complex code tasks.
- **No hardcoded token budgets**: Model generates until EOS (4096 safety net). Like Codex/OpenCode.
- **Intent routing over model-decided**: IQ3_S can't reliably pick tools. Rule-based routing (CREATE/EDIT/CHAT/RUN/SEARCH) is more reliable.

### What We Tried and Ruled Out
- **MLX**: 4-bit model (15GB) doesn't fit on 16GB without swap thrashing. No 3-bit MLX exists.
- **Speculative decode**: Slower for MoE (21.9 vs 27 tok/s). Active weights too small to benefit.
- **Multi-token prediction**: Gemma 4 not trained with MTP heads.
- **Expert pruning (top-4)**: Only 10% gain, quality risk.
- **128K context default**: OOMs under memory pressure. 32K is stable.

### Active Research
- **Kernel fusion**: Fusing map0 + matmul in Metal to eliminate 22 GPU barriers/token → potential 35-40 tok/s
- **2MB superpages**: TLB miss reduction for scattered expert reads (needs fresh reboot for contiguous memory)
- **100 tok/s goal**: Requires reducing active weight reads per token or novel compute approaches

### Testing
- Run `python tests/test_jem.py` for automated tests
- Run `python tests/test_jem.py --quick` for fast subset

### Code Style
- Minimal changes, no unnecessary refactoring
- Every feature must be tested

### MANDATORY: Before implementing ANY feature
1. **Review Codex source** at `/Users/marcsolomon/Desktop/Gemma Source - Starting Code/gem_code copy/`
2. **Review OpenCode** (Go-based) — check their approach too
3. **Take the BEST from both**, then adapt for our local-first + small model constraints
4. **Never freestyle** — always base implementations on proven patterns
5. **Make it better** — our innovations should improve on their approaches
