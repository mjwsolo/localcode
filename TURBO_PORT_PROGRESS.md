TURBO PORT PROGRESS:
Step 1: ggml.h type enums ✅ PASS
Step 2: ggml-common.h block structs ✅ PASS  
Step 3: ggml-turbo-quant.c core file ✅ PASS (compiled)
Step 4: ggml.c type traits ✅ PASS (registered, but no to_float/from_float yet)
Step 7: arg.cpp CLI flags ✅ PASS (turbo2/3/4 accepted)

REMAINING:
Step 4b: Wire to_float/from_float in type traits → needs turbo-quant.c functions declared
Step 5: Metal shaders for turbo4 → BIGGEST FILE, ~11K lines
Step 6: Metal dispatch (ggml-metal-ops.cpp) → dispatch turbo4 kernels
Step 8: llama-kv-cache.cpp → register turbo4 as valid KV cache

WORKING BUILD: /Users/marcsolomon/llama-cpp-test/build/bin/llama-server
BACKUP: /Users/marcsolomon/llama-server-stock-working
