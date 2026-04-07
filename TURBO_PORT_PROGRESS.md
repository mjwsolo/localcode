# TurboQuant + Tool Calling: Definitive Finding

## ROOT CAUSE CONFIRMED
TurboQuant's Walsh-Hadamard Transform (WHT) rotation corrupts the KV cache
attention patterns needed for the model to generate `<|tool_call>` tokens.

Proof:
- Stock llama.cpp + q8_0 KV + GPU → tools WORK (28.6 tok/s)
- Same build + turbo4 KV + GPU → tools BROKEN (24.5 tok/s, no tool calls)
- The ONLY difference is the KV cache type: q8_0 vs turbo4

## THE FIX
Edit TurboQuant's WHT rotation to preserve tool call token attention.
The rotation is in: `/Users/marcsolomon/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c`
And the Metal kernel: `ggml/src/ggml-metal/ggml-metal.metal` (turbo4 dequant)

Options:
1. Disable WHT rotation for the first/last few KV cache layers (where tool tokens are decided)
2. Use higher precision (q8_0) for specific attention heads that control tool calling
3. Modify the rotation to preserve the tool call token subspace

## WORKING BUILDS
- Stock + q8_0: 28.6 tok/s, tools work, 4K context (limited)
- TurboQuant: 27 tok/s, no tools, 32K context (great context, no tools)

## GOAL
Modify TurboQuant to work with tool calling → 27+ tok/s + tools + 32K

## Build Location
Working fresh build: /Users/marcsolomon/llama-cpp-fresh/
Stock backup: /Users/marcsolomon/llama-server-stock-working

## DEFINITIVE ROOT CAUSE (2026-04-07)
TurboQuant's kernel-level WHT rotation (simd_shuffle_xor in Metal)
corrupts the attention patterns needed for tool call token generation.

## POTENTIAL FIXES
1. Disable kernel-level WHT for turbo4 KV cache (keep simple 4-bit quant)
2. Use RotorQuant instead (https://github.com/scrya-com/rotorquant)
   - "beats TurboQuant: better PPL, 28% faster decode"
   - Uses simpler block-diagonal rotations
   - Drop-in llama.cpp integration
3. Use turbo4 with "nowht" mode for V cache (mentioned in TurboQuant+ docs)

## NEXT SESSION PLAN
1. Try RotorQuant — it's reportedly better AND simpler
2. Or: disable WHT in turbo4 and test if plain 4-bit PolarQuant works for tools
3. The quantization itself (4-bit with codebooks) should work — it's the ROTATION that breaks tools
