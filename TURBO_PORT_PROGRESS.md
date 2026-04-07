# Current State: Need Reboot Then Final Tests

## DEFINITIVE FINDINGS
1. Stock llama.cpp (fresh build) + GPU + q8_0 KV: **tools WORK, 28.6 tok/s** ✅
2. ANY TurboQuant fork build: **tools BROKEN** ❌  
3. Root cause: TurboQuant WHT rotation corrupts tool call attention
4. RotorQuant fork: also based on TurboQuant, same issue
5. System has 65MB free RAM — all results unreliable under this pressure

## WHAT WORKS RIGHT NOW
**Stock llama.cpp + q8_0 KV + GPU at 4K context: 28.6 tok/s + tools**

To get more context without TurboQuant:
- q4_0 KV gives 3.56x compression but is slow under memory pressure
- q8_0 KV gives 2x compression, 4-8K context practical

## AFTER REBOOT
1. `sudo sysctl iogpu.wired_limit_mb=14336`
2. Test stock build + q4_0 KV at 16K-32K with fresh memory
3. If q4_0 at 16K gives ~25+ tok/s, that's the shipping config
4. Then work on fixing TurboQuant rotation for tool calling

## THE REAL FIX (future)
Modify TurboQuant's quantize function to skip WHT rotation for
the first generation position (where tool call tokens are decided).
Or use RotorQuant's simpler 2D rotation that may not corrupt tools.

## BUILDS
- Stock (tools work): /Users/marcsolomon/llama-cpp-fresh/build/bin/llama-server  
- TurboQuant (fast, no tools): /Users/marcsolomon/llama-cpp-turboquant/build/bin/llama-server
- Backup: /Users/marcsolomon/llama-server-stock-working
