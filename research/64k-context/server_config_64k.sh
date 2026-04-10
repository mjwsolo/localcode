#!/bin/bash
# 64K context llama-server config
# Compare with current 32K config in CLAUDE.md

MODEL="${1:-$HOME/models/gemma-4-26b-a4b-it-IQ3_S.gguf}"

echo "=== 64K Context Test ==="
echo "Model: $MODEL"
echo ""

# Current 32K config (baseline)
echo "To run 32K baseline:"
echo "  llama-server --model $MODEL -ngl 999 --mmap -ctk q8_0 -ctv turbo4 -fa on -c 32768 --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0"
echo ""

# 64K config — double the context
echo "To run 64K test:"
echo "  llama-server --model $MODEL -ngl 999 --mmap -ctk q8_0 -ctv turbo4 -fa on -c 65536 --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0"
echo ""

echo "Expected KV cache: ~710 MiB (2x current 355 MiB)"
echo "Run benchmark_64k.py after server is up"
