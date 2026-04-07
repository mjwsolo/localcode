# DEFINITIVE: The bug is in the ggml layer

## Proof
- Replaced ALL of common/, src/llama-vocab.cpp, src/llama-chat.*, 
  src/llama-model.cpp, src/llama-model-loader.cpp from stock
- STILL no tool calls even with q8_0 KV
- Stock pure build: tools work at 27.7 tok/s
- The ONLY remaining difference: ggml/ layer (TurboQuant types + Metal kernels)

## The ggml layer change affects token generation globally
Even with q8_0 KV (no turbo compute path), the TurboQuant ggml
changes something in how the Metal compute graph runs that
prevents tool call token generation.

## NEXT: Binary search the ggml changes
The fix is in ggml/ — need to find which specific file breaks tools.
Candidates: ggml-metal.metal, ggml-metal-ops.cpp, ggml-cpu.c, ggml.c

## FALLBACK: Stock build + q8_0 KV + 8K context
27.7 tok/s + tools. Shipping this while fixing ggml.
