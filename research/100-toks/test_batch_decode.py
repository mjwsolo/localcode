"""Test: Is batch decode (4 tokens, one call) faster than 4 sequential decodes?

This tests the core thesis of GPU-autonomous decode:
If batching amortizes the 35ms overhead, we should see ~4x speedup.

NOTE: The batched version produces WRONG output (uses dummy tokens for pos 1-3).
We only care about SPEED, not correctness.
"""
import sys
sys.path.insert(0, "src")

from gem.config import load_config
from gem.runtime import GemRuntimeGateway
import time
import httpx
import json

def test_via_api():
    """Test batch vs sequential via the server API."""
    client = httpx.Client(base_url="http://127.0.0.1:8081", timeout=120)

    # Warm up
    client.post("/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })

    print("=== BATCH DECODE THESIS TEST ===\n")

    # Sequential: 20 tokens, one at a time
    t0 = time.time()
    resp = client.post("/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": "Write fibonacci in Python"}],
        "max_tokens": 20,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    t_seq = time.time() - t0
    timings = resp.json().get("timings", {})
    seq_tok_s = timings.get("predicted_per_second", 0)

    # "Batch": prompt eval of 20 tokens (this IS batched internally)
    # We send 20 tokens as prompt → they get batched → amortized overhead
    big_prompt = "The quick brown fox jumps over lazy dogs. " * 5  # ~20 tokens
    t0 = time.time()
    resp2 = client.post("/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": big_prompt}],
        "max_tokens": 1,  # just 1 decode token
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    t_batch = time.time() - t0
    timings2 = resp2.json().get("timings", {})
    batch_tok_s = timings2.get("prompt_per_second", 0)

    print(f"Sequential decode: {seq_tok_s:.1f} tok/s ({1000/seq_tok_s:.1f} ms/tok)")
    print(f"Batched prompt eval: {batch_tok_s:.1f} tok/s ({1000/batch_tok_s:.1f} ms/tok)")
    print(f"Speedup: {batch_tok_s/seq_tok_s:.1f}x")
    print()
    print("If batch >> sequential: GPU-autonomous decode WILL work.")
    print("The challenge is making decode tokens batchable (argmax on GPU).")

if __name__ == "__main__":
    test_via_api()
