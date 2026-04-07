# TurboQuant Port: Stock llama.cpp + Turbo4 KV Cache

## Status: IN PROGRESS

## What We Have
- **Working stock binary**: `/Users/marcsolomon/llama-server-stock-working`
  - Tool calling: ✅ `read_file({"path":"pyproject.toml"})` 
  - Speed: 30 tok/s on GPU, 220ms TTFT
  - Context: 4K (no TurboQuant yet)
  - Binary built from: `/Users/marcsolomon/llama-cpp-test/`

- **Working TurboQuant binary**: `/Users/marcsolomon/llama-cpp-turboquant/build/bin/llama-server`
  - Tool calling: ❌ broken
  - Speed: 28 tok/s on GPU, 270ms TTFT  
  - Context: 32K with turbo4 KV (355 MiB)

## The Goal
30 tok/s + tool calling + turbo4 32K context — all in one binary.

## Root Cause
The TurboQuant fork's bundled ggml changes break Gemma 4 tool call token generation.
Stock llama.cpp with bundled ggml works. Homebrew with system ggml works.
TurboQuant fork with bundled ggml breaks tools (even with stock chat parser files).

## Port Plan (test tools after EACH step)

### Step 1: ggml.h — Add type enums
- Add GGML_TYPE_TURBO2_0 (43), GGML_TYPE_TURBO3_0 (41), GGML_TYPE_TURBO4_0 (42)
- Add GGML_TYPE_TQ3_1S (44), GGML_TYPE_TQ4_1S (45)
- Add GGML_OP_TURBO_WHT op
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/include/ggml.h`

### Step 2: ggml-common.h — Add block structs
- Add block_turbo4, block_turbo3, block_turbo2 structs
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-common.h`

### Step 3: ggml-turbo-quant.c — Core quantize/dequantize
- Copy entire file
- Add to CMakeLists.txt
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c`

### Step 4: ggml-cpu.c — CPU backend dispatch  
- Add turbo type cases to quantize/dequantize dispatch
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-cpu/ggml-cpu.c`

### Step 5: ggml-metal.metal — Metal shaders
- Add turbo4 dequant/matmul kernels
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-metal/ggml-metal.metal`

### Step 6: Metal dispatch (ggml-metal-ops.cpp)
- Add turbo type cases to Metal dispatch
- Source: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-metal/ggml-metal-ops.cpp`

### Step 7: arg.cpp — CLI flags
- Add "turbo2", "turbo3", "turbo4" to cache type options
- Source: `/Users/marcsolomon/llama-cpp-turboquant/common/arg.cpp`

### Step 8: llama-kv-cache.cpp — KV cache registration
- Register turbo types as valid KV cache formats
- Source: `/Users/marcsolomon/llama-cpp-turboquant/src/llama-kv-cache.cpp`

## Server Launch Command (target)
```bash
/Users/marcsolomon/llama-cpp-test/build/bin/llama-server \
  --model <gguf> --port 8081 \
  -ngl 999 --mmap -ctk q8_0 -ctv turbo4 -fa on -c 32768 \
  --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0 --jinja
```

## Key Constraint
After EACH step, rebuild and test:
```bash
cmake --build build --target llama-server -j$(sysctl -n hw.ncpu)
# Then test tool calling:
curl ... tools ... | check for tool_calls
```
If tools break, REVERT that step and investigate.

## Important Files
- Stock build: `/Users/marcsolomon/llama-cpp-test/`
- TurboQuant fork: `/Users/marcsolomon/llama-cpp-turboquant/`
- Working stock binary backup: `/Users/marcsolomon/llama-server-stock-working`
- LocalCode app: `/Users/marcsolomon/github/gemma/`
