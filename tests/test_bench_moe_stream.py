import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench_moe_stream", Path(__file__).resolve().parents[1] / "scripts" / "bench_moe_stream.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

LOG = """\
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 24576.00 MiB
load_tensors:  MTL0_Mapped model buffer size =  2100.50 MiB
llama_moe_cache: MTL0 MoE cache size = 18000.00 MiB for 24576.00 MiB of host experts
llama_moe_cache: 40 layers, q8_0:   187 slots (73.0%)
llama_kv_cache:       MTL0 KV buffer size =   640.00 MiB
llama_context:       MTL0 compute buffer size =   300.25 MiB
llama_context:        CPU compute buffer size =    12.00 MiB
llama_moe_cache: ubatch <= 8: hits = 9000, misses = 1000, hit rate = 90.00%, uploaded = 5200.50 MiB
llama_moe_cache: ubatch  > 8: hits = 10, misses = 90, hit rate = 10.00%, uploaded = 80.00 MiB
"""


def test_moe_cache_lines():
    c = B.parse_moe_cache(LOG)
    assert c["hit_rate_pct"] == 90.0
    assert c["stats"]["ubatch  > 8"]["misses"] == 90
    assert c["stats"]["ubatch <= 8"]["uploaded_mib"] == 5200.5
    assert c["cache_mib"] == {"MTL0": 18000.0}
    assert not c["too_small"]
    assert B.parse_moe_cache("x\nMoE cache too small for 3 layers")["too_small"]


def test_metal_buffers_sum_only_gpu_devices():
    b = B.parse_metal_buffers(LOG)
    assert b["metal_total_mib"] == 2100.5 + 640 + 300.25 + 18000
    assert {"device": "CPU_Mapped", "kind": "model", "mib": 24576.0} in b["buffers"]


def test_bench_json():
    rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 310.5}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 21.25}]
    out = B.parse_bench_json("warming up\n" + json.dumps(rows))
    assert out == {"prefill_tok_s": {"512": 310.5}, "decode_tok_s": {"128": 21.25}}


def test_server_timings():
    r = B.parse_server_timings({"timings": {"prompt_per_second": 400.0, "predicted_per_second": 18.5, "prompt_n": 900, "predicted_n": 128}})
    assert r["decode_tok_s"] == 18.5 and r["prompt_n"] == 900
