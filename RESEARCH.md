# LocalCode Research Log

## Gemma 4 26B-A4B IQ3_S on M4 16GB — What We Know

### Performance (Proven)
- **Decode**: 27 tok/s (matches Ollama's 28.4 — near hardware ceiling)
- **TTFT**: 270ms (warm, with prompt cache)
- **Context**: 32K tokens (TurboQuant KV cache, 355 MiB)
- **Prompt eval**: 87 tok/s

### TurboQuant KV Cache
- Asymmetric: q8_0 keys + turbo4 values
- 3.8x compression: 32K context in 355 MiB (vs ~1.6GB without)
- Built from TheTom/llama-cpp-turboquant fork
- `--cache-ram 0` required — server-side prompt cache corrupts with TurboQuant WHT

### GPU Configuration
- `sysctl iogpu.wired_limit_mb=14336` raises Metal working set (resets on reboot)
- `-ngl 999 --mmap -fit off` — all layers on Metal via mmap shared buffers
- 2 graph splits (down from 22 with expert offload)
- GPU memory: ~2GB attention + mmap'd experts

### What WORKS (Proven via curl)
- System prompts (20-50 words, identity-first)
- Thinking mode (via `enable_thinking: true/false` per request)
- Code generation (150+ line pong/snake games)
- Chat responses
- **Single-turn tool calling via /completion endpoint** — model produces valid JSON tool calls

### What DOESN'T WORK
- `/v1/chat/completions` with `tools` parameter — model ignores tools entirely
- `/v1/chat/completions` with tool instructions in prompt — hallucination after cache pollution
- Multi-turn tool calling (model treats tool results as text to analyze)
- Speculative decoding (21.9 vs 27 tok/s — SLOWER for MoE)
- Multi-token prediction (Gemma 4 not trained for MTP)
- Kernel fusion (no speedup — GPU barriers are free on Apple Silicon)

### The Chat Template Problem
- `/v1/chat/completions` uses the GGUF's Jinja chat template
- The template adds thinking suppression tokens that corrupt tool-calling prompts
- `/completion` with manual Gemma 4 turn tags works perfectly
- Same finding as original memory: "Use /api/generate, NOT /api/chat"
- Gemma 4 turn format: `<bos><|turn>user\n...<turn|>\n<|turn>model\n`

### Single-Turn Tool Calling (WORKS)
```
Prompt: "<bos><|turn>user\nCall read_file with path pyproject.toml. Reply with only JSON.<turn|>\n<|turn>model\n"
Response: {"tool": "read_file", "parameters": {"path": "pyproject.toml"}}
```

### Multi-Turn Tool Calling (NEEDS RESEARCH)
- Inserting tool results as `<|turn>user\nTool result: ...<turn|>` doesn't work
- Inserting as `<|turn>tool\n...<turn|>` doesn't work either
- Need to research Gemma 4's exact expected format for tool responses
- Check Google's function calling docs for the correct multi-turn template

### Gemma 4 Benchmarks
- **Tool calling**: 85.5% tau2-bench (retail) — massive leap from Gemma 3
- **Coding**: 77.1% LiveCodeBench v6 — beats Qwen 3.5 35B
- **Reasoning**: 82.6% MMLU Pro, 88.3% AIME 2026
- Native tool tokens: `<|tool_call>call:FUNC{param:<|"|>val<|"|>}<tool_call|>`

### Google's Recommended Settings
- Temperature: 1.0 (we use 0.15 in fast mode — too low!)
- Top-p: 0.95
- Top-k: 64
- Don't include thinking from previous turns in history

### Architecture Issues
- App has multiple competing message construction paths
- compose_messages + agent_loop + app.py fallbacks all build messages differently
- Need ONE pipeline: classify → build messages → query model → parse response

### Speed Optimization Attempts
| Approach | Result |
|----------|--------|
| TurboQuant KV cache | ✅ 3.8x compression, 32K context |
| GPU full offload (sysctl) | ✅ 27 tok/s, 2 graph splits |
| Expert offload (-ot exps=CPU) | ❌ 15 tok/s, 22 graph splits |
| Speculative decode (E2B draft) | ❌ 21.9 tok/s (slower) |
| Top-4 experts | ⚠️ 29.7 tok/s (+10%, quality risk) |
| Kernel fusion (fused MoE) | ❌ No speedup |
| 2MB superpages | ❌ KERN_NO_SPACE (needs reboot) |
| Multi-token prediction | ❌ Not supported by Gemma 4 |

### Next Steps
1. Fix multi-turn tool calling (research Gemma 4 native tool response format)
2. Switch to /completion endpoint (bypass broken chat template)
3. Rebuild app message pipeline (ONE path, not three)
4. Raise temperature to 0.7+ (Google recommends 1.0)
5. Let model pick tools (85.5% accuracy — don't pre-route with regex)

### Tool Calling at IQ3_S — Detailed Findings (2026-04-07)

**Native tool calling via Ollama: WORKS but inconsistent at IQ3_S**

| Prompt | Tool Call? | Notes |
|--------|-----------|-------|
| "Read the file pyproject.toml" | ✅ YES | Direct, specific request |
| "List python files" | ❌ NO | Vague request — model thinks but doesn't call |
| "What files are in this directory?" | ❌ NO | Too vague for IQ3_S |

**The model's tool calling accuracy at IQ3_S is prompt-sensitive.**
- Direct requests ("Read X", "Run Y") → tool calls fire
- Vague requests ("List files", "What's in here?") → model thinks but doesn't call
- The 85.5% tau2 benchmark was at full precision, not IQ3_S

**Our llama-server doesn't support Gemma 4 tool calling at all.**
- The `/v1/chat/completions` `tools` parameter is ignored
- The chat template doesn't inject tool declarations correctly
- Ollama has custom `RENDERER gemma4` / `PARSER gemma4` that handles it

**Best approach for IQ3_S: Hybrid**
1. Format requests as direct tool calls in the prompt (not vague)
2. Parse model output for JSON tool calls
3. Have Python fallbacks for when tool calling doesn't fire
4. Use `/completion` endpoint with manual Gemma 4 turn tags

### THE EXACT BUG: TurboQuant Fork Broke Tool Calling (2026-04-07)

**Proven:**
- Stock homebrew llama-server (build 8660): **tool calls WORK** ✅
- Our TurboQuant fork (same model, same prompt): **tool calls BROKEN** ❌
- Not GPU-specific: broken on CPU too
- Not TurboQuant KV-specific: broken with q8_0 KV too
- The fork's modifications to `common/chat*.cpp` files broke Gemma 4 tool parsing

**The fix:**
Build from latest stock llama.cpp source and cherry-pick ONLY the TurboQuant
ggml/ files (KV cache types + Metal kernels). The chat/tool parsing code
should come from stock llama.cpp, not the TurboQuant fork.

**TurboQuant-specific files to port:**
- `ggml/src/ggml-turbo-quant.c` — core TQ quantization
- `ggml/src/ggml-metal/ggml-metal.metal` — Metal kernel additions for turbo types
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — Metal dispatch for turbo types
- Various header changes in `ggml/include/ggml.h` for type enums
- CMakeLists changes to compile TQ files

**NOT needed from fork:**
- `common/chat*.cpp` — use stock versions (they work for tools)
- `common/jinja/` — use stock
- Our custom patches (mmap bloat detection etc.) — re-apply on stock base

### BREAKTHROUGH: Homebrew Works, Our Build Doesn't (2026-04-07 late)

| Build | Tools | Speed | Context | GPU |
|-------|-------|-------|---------|-----|
| **Homebrew ngl=999 4K** | ✅ | **30.6 tok/s** | 4K | ✅ |
| Homebrew ngl=999 q4_0 32K | ✅ | 13.1 tok/s | 32K | ✅ |
| TurboQuant fork turbo4 32K | ❌ | 28.8 tok/s | 32K | ✅ |
| TurboQuant fork q8_0 CPU | ❌ | ~18 tok/s | 10K | ❌ |

**Root cause: Homebrew uses `-DLLAMA_USE_SYSTEM_GGML=ON`.**
It links against `brew install ggml` (stock ggml) instead of the bundled ggml.
Our builds use bundled ggml which has TurboQuant modifications that break
tool call token generation.

**Fix: Build with system ggml + add TurboQuant types to system ggml.**
Or: just use homebrew binary with q4_0 KV (tools work, 13 tok/s at 32K).
Or: add TurboQuant to homebrew's ggml package.

**For now: use homebrew binary at 4K-8K context (30 tok/s + tools).**
TurboQuant 32K context can come later once ggml integration is fixed.

### SOLUTION FOUND: Stock llama.cpp + Surgical TurboQuant Port (2026-04-07)

**CONFIRMED: Fresh stock llama.cpp from source with bundled ggml + GPU = tool calling WORKS.**
The earlier "stock build that failed" had contaminated files from the fork.

**Working binary saved:** `/Users/marcsolomon/llama-server-stock-working`
- Tools: ✅ `read_file({"path":"pyproject.toml"})`
- Speed: ~30 tok/s on GPU
- Context: 4K (no TurboQuant KV yet)

**Plan: Add turbo4 KV cache to stock build, one file at a time:**
1. ggml.h — add GGML_TYPE_TURBO4_0 enum
2. ggml-common.h — add block_turbo4 struct
3. ggml-turbo-quant.c — core quantize/dequantize
4. ggml-cpu.c — CPU backend dispatch
5. ggml-metal.metal — Metal kernels
6. arg.cpp — CLI flag
7. llama-kv-cache.cpp — registration
8. Test after EACH step — tool calling must keep working

**The goal: 30 tok/s + tool calling + turbo4 32K context**
