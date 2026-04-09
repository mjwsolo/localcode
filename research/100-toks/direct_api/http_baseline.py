#!/usr/bin/env python3
"""HTTP server baseline measurement for comparison with direct API."""
import httpx
import time
import json

client = httpx.Client(base_url="http://127.0.0.1:8081", timeout=120)

# Warmup
print("Warming up...")
client.post("/v1/chat/completions", json={
    "model": "gemma",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 5,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
})

prompt = "Write a Python function to compute fibonacci numbers efficiently."
N_TOKENS = 30

print(f"\nStreaming {N_TOKENS} tokens via HTTP server...")

# Run 3 times for stability
for run in range(3):
    t0 = time.perf_counter()
    token_times = []
    text = ""
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": N_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }) as resp:
        last_t = t0
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                now = time.perf_counter()
                token_times.append((now - last_t) * 1000)
                last_t = now
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    text += delta
                except:
                    pass

    t_total = time.perf_counter() - t0

    if len(token_times) > 2:
        ttft = token_times[0]
        decode_times = token_times[1:]
        avg_decode = sum(decode_times) / len(decode_times)
        http_tps = 1000.0 / avg_decode if avg_decode > 0 else 0

        print(f"\nRun {run+1}:")
        print(f"  TTFT: {ttft:.1f}ms")
        print(f"  Decode tokens: {len(decode_times)}")
        print(f"  Avg per-token: {avg_decode:.2f}ms = {http_tps:.1f} tok/s")
        print(f"  Min/Max: {min(decode_times):.2f}/{max(decode_times):.2f}ms")
        print(f"  Total: {t_total*1000:.0f}ms")

        if run == 0:
            print(f"\n  Per-token breakdown (first 15):")
            for i, t in enumerate(decode_times[:15]):
                print(f"    Token {i+1}: {t:.2f}ms = {1000/t:.1f} tok/s")

        if text:
            print(f"  Text: {repr(text[:100])}")
