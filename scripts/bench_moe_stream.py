#!/usr/bin/env python3
"""Benchmark MoE expert streaming: decode and prefill tok/s, expert-cache hit
rate and Metal buffer sizes, as JSON.

Two modes, both against binaries you point at (nothing is downloaded):

  bench_moe_stream.py bench  --bin PATH/llama-bench  --model M.gguf [--label L] -- <extra llama-bench flags>
  bench_moe_stream.py server --bin PATH/llama-server --model M.gguf [--label L] [--ctx N] -- <extra llama-server flags>

`bench` runs llama-bench with -o json (pass `-ot exps=CPU --moe-cache-mib N`
after `--` to benchmark the cache) and records prompt/generation tok/s.
`server` starts llama-server, sends one completion, reads the server's own
timings, stops it and parses the log (cache stats are printed at shutdown).
Run it only when the GPU is free. The parsers are plain functions over text
and are unit-tested with canned logs (tests/test_bench_moe_stream.py).
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CACHE_RE = re.compile(
    r"llama_moe_cache: (?P<kind>[^:]+): hits = (?P<hits>\d+), misses = (?P<misses>\d+), "
    r"hit rate = (?P<rate>[\d.]+)%, uploaded = (?P<mib>[\d.]+) MiB")
BUFFER_RE = re.compile(r"(?P<dev>[A-Za-z0-9_]+)\s+(?P<kind>[A-Za-z][\w ]*?)\s*buffer size =\s*(?P<mib>[\d.]+) MiB")
CACHE_SIZE_RE = re.compile(r"(?P<dev>\w+)\s+MoE cache size =\s*(?P<mib>[\d.]+) MiB for (?P<experts>[\d.]+) MiB of host experts")


def parse_bench_json(text: str) -> dict:
    """llama-bench `-o json`: {"prefill_tok_s": {n_prompt: tok/s}, "decode_tok_s": {n_gen: tok/s}}."""
    start = text.find("[")
    rows = json.loads(text[start:]) if start >= 0 else []
    out: dict = {"prefill_tok_s": {}, "decode_tok_s": {}}
    for r in rows:
        ts = r.get("avg_ts")
        if ts is None:
            continue
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            out["prefill_tok_s"][str(r["n_prompt"])] = ts
        elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            out["decode_tok_s"][str(r["n_gen"])] = ts
    return out


def parse_moe_cache(log: str) -> dict:
    """Cache stats lines. Returns {"stats": {kind: {...}}, "hit_rate": decode-batch rate or None,
    "cache_mib": {device: MiB}, "too_small": bool}. When the server logs stats more
    than once (periodic), the last line per kind wins."""
    stats: dict = {}
    for m in CACHE_RE.finditer(log):
        stats[m["kind"].strip()] = {"hits": int(m["hits"]), "misses": int(m["misses"]),
                                    "hit_rate_pct": float(m["rate"]), "uploaded_mib": float(m["mib"])}
    cache_mib = {m["dev"]: float(m["mib"]) for m in CACHE_SIZE_RE.finditer(log)}
    small = stats.get("ubatch <= 8")
    return {"stats": stats, "hit_rate_pct": small["hit_rate_pct"] if small else None,
            "cache_mib": cache_mib, "too_small": "MoE cache too small" in log}


def parse_metal_buffers(log: str) -> dict:
    """Buffer sizes per device from the load log. ``metal_total_mib`` sums every
    buffer on an MTL* device (what counts against the GPU working set), including the
    expert cache bank; CPU buffers are listed but not summed."""
    buffers = []
    for m in BUFFER_RE.finditer(log):
        buffers.append({"device": m["dev"], "kind": m["kind"].strip(), "mib": float(m["mib"])})
    metal = sum(b["mib"] for b in buffers if b["device"].startswith("MTL"))
    metal += sum(float(m["mib"]) for m in CACHE_SIZE_RE.finditer(log) if m["dev"].startswith("MTL"))
    return {"buffers": buffers, "metal_total_mib": round(metal, 2)}


def parse_server_timings(response: dict) -> dict:
    t = response.get("timings") or {}
    return {"prefill_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
            "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n")}


def run_bench(binary: str, model: str, extra: list[str]) -> dict:
    cmd = [binary, "-m", model, "-o", "json", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({p.returncode}): {p.stderr[-500:]}")
    result = {"cmd": cmd, **parse_bench_json(p.stdout)}
    result["load_log"] = {**parse_metal_buffers(p.stderr), **parse_moe_cache(p.stderr)}
    return result


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(binary: str, model: str, ctx: int, extra: list[str], prompt_words: int = 1500, n_predict: int = 128) -> dict:
    port = _free_port()
    cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(ctx), *extra]
    log_path = Path(f"bench-moe-{port}.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError("llama-server exited during load; see " + str(log_path))
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                    break
                except Exception:  # noqa: BLE001 - still loading
                    time.sleep(1)
            body = json.dumps({"prompt": "Summarise: " + "alpha beta gamma " * (prompt_words // 3),
                               "n_predict": n_predict, "temperature": 0, "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
            response = json.load(urllib.request.urlopen(req, timeout=900))
        finally:
            proc.send_signal(signal.SIGINT)  # cache stats are logged on shutdown
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
    text = log_path.read_text(errors="replace")
    return {"cmd": cmd, **parse_server_timings(response), **parse_metal_buffers(text), **parse_moe_cache(text)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bench", "server"])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", help="write JSON here as well as stdout")
    a = ap.parse_args(argv)
    result = run_bench(a.bin, a.model, extra) if a.mode == "bench" else run_server(a.bin, a.model, a.ctx, extra)
    result = {"label": a.label, "mode": a.mode, **result}
    text = json.dumps(result, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
