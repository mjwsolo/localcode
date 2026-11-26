"""Expert-tensor sizes of a MoE GGUF, read from the header alone.

Sizes come from tensor offsets (next offset minus this one), so no ggml type
table is needed; alignment padding is at most 31 bytes per tensor. A split GGUF
is read shard by shard. Used by fit.py to decide whether a MoE model can stream
its experts from the page cache instead of holding them in the GPU working set.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_exps\.")

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass(frozen=True)
class MoeInfo:
    expert_bytes: int      # all ffn_{gate,up,down,gate_up}_exps tensors
    n_expert: int
    n_expert_used: int
    n_moe_layers: int      # layers that carry expert tensors

    @property
    def per_expert_bytes(self) -> int:
        """One expert's slot in the cache bank: its gate, up and down together."""
        return self.expert_bytes // max(1, self.n_expert * self.n_moe_layers)

    @property
    def min_bank_bytes(self) -> int:
        """Smallest bank that can hold the experts one token selects in every layer."""
        return self.n_expert_used * self.n_moe_layers * self.per_expert_bytes


def _read_shard(path: Path) -> tuple[dict[str, int], int, set[int]]:
    """({n_expert, n_expert_used}, expert bytes, expert layer indexes) of one file."""
    meta: dict[str, int] = {}
    tensors: list[tuple[int, str]] = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        f.read(4)
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        alignment = 32

        def rs() -> str:
            ln, = struct.unpack("<Q", f.read(8))
            return f.read(ln).decode("utf-8", "replace")

        def rv(t: int):
            if t == 8:
                return rs()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                ln, = struct.unpack("<Q", f.read(8))
                if et in _SCALAR and et != 7:
                    f.seek(ln * struct.calcsize(_SCALAR[et]), 1)
                    return None
                return [rv(et) for _ in range(ln)]
            fmt = _SCALAR[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            k = rs()
            t, = struct.unpack("<I", f.read(4))
            v = rv(t)
            if isinstance(v, int):
                if k.endswith(".expert_count"):
                    meta["n_expert"] = v
                elif k.endswith(".expert_used_count"):
                    meta["n_expert_used"] = v
                elif k == "general.alignment":
                    alignment = v or 32
        for _ in range(n_tensors):
            name = rs()
            n_dims, = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims + 4)
            offset, = struct.unpack("<Q", f.read(8))
            tensors.append((offset, name))
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
    end = path.stat().st_size - data_start
    tensors.sort()
    total = 0
    layers: set[int] = set()
    for i, (offset, name) in enumerate(tensors):
        m = EXPERT_TENSOR_RE.match(name)
        if not m:
            continue
        nxt = tensors[i + 1][0] if i + 1 < len(tensors) else end
        total += max(0, nxt - offset)
        layers.add(int(m.group(1)))
    return meta, total, layers


def read_moe_info(first: Path | str) -> MoeInfo | None:
    """Expert byte total and shape of the model whose first (or only) file is
    ``first``; None for a dense model or an unreadable file."""
    from .split_gguf import shard_paths
    try:
        total = 0
        layers: set[int] = set()
        meta: dict[str, int] = {}
        for p in shard_paths(Path(first)):
            m, size, ls = _read_shard(p)
            meta = meta or m
            for k, v in m.items():
                meta.setdefault(k, v)
            total += size
            layers |= ls
        if not total or not layers or not meta.get("n_expert") or not meta.get("n_expert_used"):
            return None
        return MoeInfo(total, meta["n_expert"], meta["n_expert_used"], len(layers))
    except (OSError, ValueError, struct.error, KeyError):
        return None
