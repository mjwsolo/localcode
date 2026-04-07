# SOLVED: TurboQuant + Tool Calling + 32K Context

## Final Config
- **Speed**: 24-26 tok/s decode, 223ms TTFT
- **Tools**: ✅ Native Gemma 4 tool calling working
- **Context**: 32K tokens with TurboQuant turbo4 KV cache (355 MiB)
- **GPU**: Full Metal offload (ngl=999)

## The Fix
Cherry-picked upstream llama.cpp's Gemma 4 fixes into the TurboQuant fork:
- `common/chat*.cpp/h` — updated chat parser with tool call support
- `src/llama-vocab.cpp` — Gemma 4 BOS token fix (PR #21500)
- `src/llama-chat.cpp/h` — updated template handling

## Binary
`/Users/marcsolomon/llama-cpp-turboquant/build/bin/llama-server`

## Launch Command
```bash
sudo sysctl iogpu.wired_limit_mb=14336

llama-server \
  --model <gguf> --port 8081 \
  -ngl 999 --mmap -ctk q8_0 -ctv turbo4 -fa on -c 32768 \
  --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0
```
