# TurboQuant Port: Status

## Current State (needs reboot to verify)
System is under severe memory pressure (65MB free). Tool calling results
are unreliable because GPU memory pressure affects model output quality.

## What We Know For Certain
- Homebrew llama-server with ngl=999 on a FRESH system: tools WORK (30 tok/s)
- Ollama with same model: tools WORK (28 tok/s)
- Our builds: tools inconsistent (may be memory pressure, not code)

## What Needs Testing After Reboot
1. Stock build from source on GPU → does tool calling work?
2. If yes: add TurboQuant ggml → still work?
3. If yes: done! We have tools + turbo + 32K

## Files Ready
- Stock build: /Users/marcsolomon/llama-cpp-test/ (has fork's ggml + src + tools, stock common/)
- Stock backup binary: /Users/marcsolomon/llama-server-stock-working
- Homebrew binary: /opt/homebrew/bin/llama-server (known working for tools)

## Server Config (use after reboot)
```bash
sudo sysctl iogpu.wired_limit_mb=14336

# Test with homebrew first (known working):
/opt/homebrew/bin/llama-server \
  --model <gguf> --port 8081 \
  -ngl 999 --mmap -fa on -c 4096 \
  --threads 10 -np 1 -fit off --cache-ram 0

# Then test stock from source:
/Users/marcsolomon/llama-cpp-test/build/bin/llama-server \
  --model <gguf> --port 8081 \
  -ngl 999 --mmap -ctk q8_0 -ctv turbo4 -fa on -c 32768 \
  --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0
```

## Tool Call Test Command
```bash
curl -s http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Read the file pyproject.toml"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"max_tokens":200,"temperature":0.7,"chat_template_kwargs":{"enable_thinking":true}}'
```
