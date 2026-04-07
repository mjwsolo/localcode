# TurboQuant Port Progress

## FINDING: The bug is in BUNDLED ggml, not fork changes

ANY build with bundled ggml (stock OR fork) breaks tool calling on GPU.
The homebrew binary works because it uses SYSTEM ggml (-DLLAMA_USE_SYSTEM_GGML=ON).

The difference is between:
- Homebrew ggml 0.9.11 (system package) → tools WORK
- Any bundled ggml (stock from source) → tools BROKEN on GPU

This is NOT a TurboQuant issue. It's a ggml build issue.

## Next Step
Either:
1. Build with -DLLAMA_USE_SYSTEM_GGML=ON (requires installing ggml with turbo types)
2. Find the specific ggml build difference that breaks GPU tool calling
3. Use homebrew binary for now (tools work, no TurboQuant, 8K context)

## Current Best Config
Homebrew binary: 30.6 tok/s, tools work, 4-8K context
TurboQuant binary: 27 tok/s, no tools, 32K context
