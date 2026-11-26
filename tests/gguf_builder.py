"""Write tiny GGUF headers with sparse tensor data, for header-reader tests."""
from __future__ import annotations

import struct
from pathlib import Path


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, tensors: list[tuple[str, int]], kv: dict[str, int] | None = None, alignment: int = 32) -> Path:
    """``tensors`` is [(name, byte size)]; the data section is a sparse file."""
    kv = kv or {}
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1)
    head += _s("general.architecture") + struct.pack("<I", 8) + _s("qwen3moe")
    for k, v in kv.items():
        head += _s(k) + struct.pack("<II", 4, v)
    infos = b""
    offset = 0
    for name, size in tensors:
        infos += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", size) + struct.pack("<IQ", 0, offset)
        offset += (size + alignment - 1) // alignment * alignment
    pos = len(head) + len(infos)
    data_start = (pos + alignment - 1) // alignment * alignment
    with open(path, "wb") as f:
        f.write(head + infos + b"\0" * (data_start - pos))
        f.truncate(data_start + offset)
    return path


def moe_tensors(n_layers: int, expert_bytes_per_layer: int, dense_bytes: int = 4096) -> list[tuple[str, int]]:
    out = [("token_embd.weight", dense_bytes)]
    for i in range(n_layers):
        out += [(f"blk.{i}.attn_q.weight", dense_bytes),
                (f"blk.{i}.ffn_gate_exps.weight", expert_bytes_per_layer),
                (f"blk.{i}.ffn_up_exps.weight", expert_bytes_per_layer),
                (f"blk.{i}.ffn_down_exps.weight", expert_bytes_per_layer)]
    return out
