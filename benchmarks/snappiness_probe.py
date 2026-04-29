"""Snappiness probe — sweep llama-server configs and measure decode + system.

Runs the same fixed prompt through each (threads, priority, sysctl_cap) combo,
captures:
  - decode tok/s
  - prompt eval tok/s
  - peak wired memory
  - compressor growth during decode (proxy for "system under pressure")
  - laptop-responsiveness proxy: spin 2 CPU-bound background loops in Python,
    measure how many iterations they complete during the decode window.
    Higher iterations = more CPU left for UI/other apps.

Writes a markdown table to stdout + benchmarks/results/snappiness.md.

Usage:
    python benchmarks/snappiness_probe.py \
        --model ~/.local/share/localcode/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
        [--runs 3]

Requires sudo for sysctl tests (script will prompt if those are enabled).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA_BIN = Path.home() / "llama-cpp-turboquant" / "build" / "bin" / "llama-server"
PORT = 8181  # avoid 8081 in case the main server is up
URL = f"http://127.0.0.1:{PORT}"

FIXED_PROMPT = (
    "Write a Python function `merge_sort(arr)` that implements merge sort. "
    "Include a docstring and handle the empty-list case. "
    "After the function, add a doctest-style block showing 3 calls."
)


def _vmstat() -> dict[str, int]:
    """Parse vm_stat into a dict of metric → pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        try:
            d[k.strip()] = int(v)
        except ValueError:
            pass
    return d


def _page_gb(pages: int) -> float:
    return pages * 16384 / 1073741824  # 16 KB pages on Apple Silicon


def launch_server(model: str, threads: int, nice_level: int = 0,
                  ctx: int = 8192) -> subprocess.Popen:
    cmd = []
    if nice_level:
        cmd = ["nice", "-n", str(nice_level)]
    cmd += [
        str(LLAMA_BIN),
        "--model", model,
        "-ngl", "999", "--mmap",
        "-ctk", "q8_0", "-ctv", "turbo4",
        "-fa", "on", "-c", str(ctx),
        "--threads", str(threads),
        "-b", "2048", "-ub", "512",
        "-np", "1", "-fit", "off", "--cache-ram", "0",
        "--port", str(PORT),
    ]
    log = open("/tmp/snappiness_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    # Wait for health
    for _ in range(90):
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=2).read()
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("server never became healthy")


def shutdown(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


# Background CPU workers — measure headroom via separate Python processes
# (bypass GIL). Each worker counts loop iterations until it sees a stop file,
# then prints the count and exits. More iterations = more CPU left for other
# apps during inference, which is the "laptop feels snappy" proxy.
_WORKER_SRC = """
import os, sys, time
stop = sys.argv[1]
count = 0
while not os.path.exists(stop):
    x = [i * 3 for i in range(100)]
    x.sort()
    count += 1
sys.stdout.write(str(count))
sys.stdout.flush()
"""


def probe(label: str) -> dict:
    """Send fixed prompt, return timings + memory deltas + headroom count."""
    stop_file = f"/tmp/snapprobe_stop_{os.getpid()}"
    try:
        os.remove(stop_file)
    except FileNotFoundError:
        pass

    # Spawn 2 CPU-bound subprocess workers.
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER_SRC, stop_file],
            stdout=subprocess.PIPE,
        )
        for _ in range(2)
    ]

    pre = _vmstat()
    # Do the inference.
    t0 = time.time()
    payload = {
        "messages": [
            {"role": "system", "content": "Python coding assistant."},
            {"role": "user", "content": FIXED_PROMPT},
        ],
        "temperature": 0.0, "max_tokens": 500, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    infer_elapsed = time.time() - t0

    # Stop workers and collect iteration counts.
    open(stop_file, "w").close()
    headroom = 0
    for w in workers:
        out, _ = w.communicate(timeout=5)
        try:
            headroom += int(out.decode().strip())
        except Exception:
            pass

    try:
        os.remove(stop_file)
    except FileNotFoundError:
        pass
    post = _vmstat()

    usage = resp.get("usage", {}) or {}
    # llama-server also returns `timings` block
    # We'll re-query /timings or trust the decode rate from response
    # chat completions doesn't include our rich timings; use a direct /completion probe after
    t_decode = 0.0
    t_prompt = 0.0
    try:
        t_req = urllib.request.Request(
            f"{URL}/completion",
            data=json.dumps({"prompt": "def factorial(n):",
                             "n_predict": 80, "temperature": 0.0,
                             "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        tjson = json.loads(urllib.request.urlopen(t_req, timeout=60).read())
        t_decode = tjson["timings"]["predicted_per_second"]
        t_prompt = tjson["timings"]["prompt_per_second"]
    except Exception:
        pass

    return {
        "label": label,
        "infer_wall_s": infer_elapsed,
        "chat_completion_tokens": usage.get("completion_tokens"),
        "chat_prompt_tokens": usage.get("prompt_tokens"),
        "decode_tok_s": t_decode,
        "prompt_tok_s": t_prompt,
        "wired_pre_gb": _page_gb(pre.get("Pages wired down", 0)),
        "wired_post_gb": _page_gb(post.get("Pages wired down", 0)),
        "compressor_pre_gb": _page_gb(pre.get("Pages occupied by compressor", 0)),
        "compressor_post_gb": _page_gb(post.get("Pages occupied by compressor", 0)),
        "swapins_delta": post.get("Swapins", 0) - pre.get("Swapins", 0),
        "headroom_iters": headroom,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    # Test matrix — all preserve the "localcode runs alongside VSCode" workflow.
    # No quantization changes (we're going UP in quant, not down).
    # sysctl changes are gated behind sudo — skipped here; run manually if
    # you want to sweep iogpu.wired_limit_mb.
    configs = [
        # (label, threads, nice_level)
        ("threads=10, nice=0 (baseline)", 10, 0),
        ("threads=8,  nice=0",             8, 0),
        ("threads=6,  nice=0",             6, 0),
        ("threads=10, nice=10",           10, 10),
        ("threads=6,  nice=10",            6, 10),
        ("threads=4,  nice=10",            4, 10),
    ]

    results = []
    for label, threads, nice in configs:
        print(f"\n=== {label} — launching... ===")
        # Make sure nothing else is on the port
        subprocess.run(["pkill", "-f", f"llama-server.*--port {PORT}"],
                       capture_output=True)
        time.sleep(1)
        try:
            proc = launch_server(args.model, threads, nice, args.ctx)
        except Exception as e:
            print(f"  FAILED to launch: {e}")
            results.append({"label": label, "error": str(e)})
            continue
        try:
            rows = [probe(label) for _ in range(args.runs)]
            if args.runs > 1:
                # Average numeric fields
                agg = {"label": label}
                keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
                for k in keys:
                    agg[k] = sum(r[k] for r in rows) / len(rows)
                results.append(agg)
            else:
                results.append(rows[0])
            r = results[-1]
            print(f"  decode={r.get('decode_tok_s',0):.1f} tok/s  "
                  f"prompt={r.get('prompt_tok_s',0):.1f} tok/s  "
                  f"wired+={r.get('wired_post_gb',0) - r.get('wired_pre_gb',0):+.2f} GB  "
                  f"headroom_iters={r.get('headroom_iters',0)}  "
                  f"swapins={r.get('swapins_delta',0):,}")
        finally:
            shutdown(proc)
            time.sleep(2)

    # Write markdown table
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "snappiness.md"
    with out.open("w") as f:
        f.write("# Snappiness probe\n\n")
        f.write("Higher `decode tok/s` = faster inference. "
                "Higher `headroom_iters` = more CPU left for system/UI during decode.\n\n")
        f.write("| config | decode tok/s | prompt tok/s | wired Δ (GB) | headroom iters | swapins during probe |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['label']} | ERROR: {r['error']} | | | | |\n")
                continue
            wired_delta = r.get('wired_post_gb', 0) - r.get('wired_pre_gb', 0)
            f.write(f"| {r['label']} "
                    f"| {r.get('decode_tok_s', 0):.1f} "
                    f"| {r.get('prompt_tok_s', 0):.1f} "
                    f"| {wired_delta:+.2f} "
                    f"| {int(r.get('headroom_iters', 0)):,} "
                    f"| {int(r.get('swapins_delta', 0)):,} |\n")
    print(f"\nResults written → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
