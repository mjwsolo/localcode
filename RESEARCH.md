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
