"""List every GGUF quant a HuggingFace repo ships, with sizes + a fit badge.

This module is ADDITIVE — it powers a HuggingFace-style "browse all quants"
picker on top of the curated ``CHOICES`` list. It does NOT touch the existing
download/runtime path (``ModelChoice``, ``download_model``, ``recommend`` …);
it only *reads* a repo's file tree from the public HF API and reports sizes.

Design goals:
  * dependency-light: std-lib + httpx (already a project dependency).
  * never raise on network/HTTP failure — degrade to a stale disk cache, else [].
  * sizes are DECIMAL GB (bytes / 1e9), matching the catalog's HF convention.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .split_gguf import display_alias, parse_shard

# HF tree API: lists every file in the repo (recursive). LFS files (the .gguf
# weights) carry their real size under entry["lfs"]["size"]; small files use
# entry["size"]. We read both defensively.
_TREE_URL = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
_CACHE_TTL_SEC = 24 * 60 * 60  # treat cache fresh for 24h
_TIMEOUT_SEC = 15.0

# Parse the quant tag out of a filename. Unsloth names look like:
#   gemma-4-12b-it-UD-Q4_K_XL.gguf  ->  UD-Q4_K_XL
#   Qwen3.6-35B-A3B-Q8_0.gguf       ->  Q8_0
#   ...-BF16.gguf / ...-IQ2_M.gguf  ->  BF16 / IQ2_M
# We grab the trailing quant token: an optional UD- prefix, then a quant code
# (Q/IQ/BF/F + digits + optional _K / _XL / _S / _M / _0 / _1 suffixes).
_QUANT_RE = re.compile(
    r"(UD-)?((?:IQ|Q|BF|F)\d+(?:_[A-Z0-9]+)*(?:-mix)?)\.gguf$", re.IGNORECASE
)


def _home_dir() -> Path:
    """(LOCALCODE_HOME or ~/.localcode) — mirrors config.get_home_dir()."""
    override = os.environ.get("LOCALCODE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".localcode"


def _cache_path(hf_repo: str) -> Path:
    slug = hf_repo.replace("/", "__")  # "unsloth/foo-GGUF" -> "unsloth__foo-GGUF"
    return _home_dir() / "cache" / "hf_quants" / f"{slug}.json"


def _parse_label(filename: str) -> str | None:
    """Quant tag from a .gguf filename (e.g. 'UD-Q4_K_XL'), or None when the
    name has no recognizable quant code.

    Returning None is how we DROP non-weight files (MTP drafters, stray
    `…-it.gguf`, etc.) instead of inventing junk labels like 'it' or 'MTP'.
    """
    base = filename.rsplit("/", 1)[-1]  # drop any directory prefix
    if parse_shard(base) is not None:  # split quant: label comes from the stem
        base = display_alias(base) + ".gguf"
    m = _QUANT_RE.search(base)
    if m:
        return (m.group(1) or "") + m.group(2)
    return None


@dataclass(frozen=True)
class Quant:
    filename: str       # repo-relative path of the .gguf, e.g. "gemma-4-12b-it-Q8_0.gguf"
    size_gb: float      # decimal GB (bytes / 1e9)
    label: str          # parsed quant tag, e.g. "UD-Q4_K_XL", "Q8_0", "BF16", "IQ2_M"
    is_mmproj: bool     # True for vision projector sidecars (mmproj-*.gguf)
    # Split (sharded) quant: ((repo_path, bytes), ...) in shard order. Then
    # ``filename`` is the FIRST shard's basename (what llama-server loads, stored
    # flat in the models dir) and ``size_gb`` is the sum of all shards.
    shards: tuple[tuple[str, int], ...] = ()


def _quant_from_entry(entry: dict) -> Quant | None:
    """Build a Quant from one HF tree entry, or None if it's not a selectable
    model-weight quant.

    The browser lists things the user PICKS. It must show only real model-weight
    quants — never sidecar files, which are tiny and confusing next to a
    multi-GB model (a 12B repo showing "BF16 · 0.9 GB" rows is the vision
    projector, not a 12B model). Dropped here:
      - vision projectors (`mmproj-*.gguf`) — auto-paired with the chosen quant
        via the catalog's mmproj fields, never selected directly
      - speculative-decoding draft heads (`mtp-*.gguf`) and anything in a
        subfolder (e.g. `MTP/…`) — not standalone models
      - any file with no recognizable quant code (`…-it.gguf` extras)
    """
    path = entry.get("path", "")
    if not path.lower().endswith(".gguf"):
        return None
    base = path.rsplit("/", 1)[-1]
    lowered = base.lower()
    # Subfolder files (e.g. `MTP/gemma-…-Q4_0.gguf`) are never top-level
    # selectable weights.
    if "/" in path:
        return None
    # Shards are grouped into one split quant by _split_quants.
    if parse_shard(base) is not None:
        return None
    # Vision projectors and speculative drafters (MTP heads, DFlash) are
    # sidecars, not pickable models.
    if lowered.startswith(("mmproj", "mtp-", "mtp_", "dflash")):
        return None
    label = _parse_label(path)
    # Real weight quants carry a quant code (Q4_K_M, IQ3_S, BF16, …). Anything
    # else is repo noise — drop it so the picker never shows junk rows.
    if label is None:
        return None
    is_mmproj = False
    # LFS weights store the true size under entry["lfs"]["size"]; fall back to
    # the top-level "size" for non-LFS files.
    lfs = entry.get("lfs") or {}
    size_bytes = lfs.get("size") or entry.get("size") or 0
    return Quant(
        filename=path,
        size_gb=round(size_bytes / 1e9, 3),
        label=label,
        is_mmproj=is_mmproj,
    )


_SIDECAR_PREFIXES = ("mmproj", "mtp-", "mtp_", "dflash")


def _entry_bytes(entry: dict) -> int:
    lfs = entry.get("lfs") or {}
    return int(lfs.get("size") or entry.get("size") or 0)


def _split_quants(entries: list[dict]) -> list[Quant]:
    """Group shard files (``<stem>-00001-of-00003.gguf``, top level or in a
    subfolder) into one Quant per split. A split is listed only when every shard
    is present. Sidecars (mmproj, mtp, dflash) are skipped by file or folder name."""
    groups: dict[tuple[str, str, int], dict[int, dict]] = {}
    for entry in entries:
        path = entry.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        parts = path.split("/")
        if parts[-1].lower().startswith(_SIDECAR_PREFIXES) or any(
            folder.lower().startswith(("mmproj", "mtp", "dflash")) for folder in parts[:-1]
        ):
            continue
        parsed = parse_shard(parts[-1])
        if parsed is None:
            continue
        stem, idx, total = parsed
        groups.setdefault(("/".join(parts[:-1]), stem, total), {})[idx] = entry
    out: list[Quant] = []
    for (_, stem, total), found in groups.items():
        if sorted(found) != list(range(1, total + 1)):
            continue
        label = _parse_label(stem + ".gguf")
        if label is None:
            continue
        shards = tuple((found[i]["path"], _entry_bytes(found[i])) for i in range(1, total + 1))
        out.append(Quant(
            filename=found[1]["path"].rsplit("/", 1)[-1],
            size_gb=round(sum(b for _, b in shards) / 1e9, 3),
            label=label,
            is_mmproj=False,
            shards=shards,
        ))
    return out


def _read_cache(hf_repo: str) -> tuple[list[Quant], float] | None:
    """Return (quants, fetched_at) from disk, or None if no/invalid cache."""
    p = _cache_path(hf_repo)
    try:
        raw = json.loads(p.read_text())
        quants = [Quant(**{**q, "shards": tuple((str(a), int(b)) for a, b in q.get("shards") or ())})
                  for q in raw["quants"]]
        return quants, float(raw.get("fetched_at", 0.0))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(hf_repo: str, quants: list[Quant]) -> None:
    p = _cache_path(hf_repo)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": time.time(), "quants": [asdict(q) for q in quants]}
        p.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass  # caching is best-effort; never fail the caller over it


def fetch_quants(hf_repo: str) -> list[Quant]:
    """List the GGUF quants of ``hf_repo`` (sorted by size ascending).

    Uses a 24h disk cache. Network/HTTP errors NEVER raise — we fall back to
    the on-disk cache if present, otherwise return []. mmproj-*.gguf sidecars
    are included and flagged via ``Quant.is_mmproj``.
    """
    cached = _read_cache(hf_repo)
    # Fresh cache wins outright — no network call needed.
    if cached and (time.time() - cached[1]) < _CACHE_TTL_SEC:
        return cached[0]

    try:
        resp = httpx.get(
            _TREE_URL.format(repo=hf_repo),
            timeout=_TIMEOUT_SEC,
            follow_redirects=True,
            headers={"User-Agent": "localcode/hf_quants"},
        )
        resp.raise_for_status()
        entries = resp.json()
    except (httpx.HTTPError, ValueError):
        # Any network/HTTP/JSON failure: serve stale cache if we have one, else [].
        return cached[0] if cached else []

    quants = [q for e in entries if (q := _quant_from_entry(e)) is not None]
    quants += _split_quants(entries)
    quants.sort(key=lambda q: q.size_gb)
    _write_cache(hf_repo, quants)
    return quants


def fit_badge(size_gb: float, ram_gb: int, filename: str | None = None) -> str:
    """Fit badge for a quant whose files total ``size_gb`` on a Mac with ``ram_gb``.

    Same rule as models_catalog.recommend() and the launch check (localcode.fit):
    files plus the model's measured run-time memory must sit inside the GPU's
    working set. Returns "fits" | "tight" | "too_big".
    """
    from .fit import fit
    return fit(size_gb, ram_gb, filename)


if __name__ == "__main__":  # pragma: no cover - tiny self-test
    # Badge logic on a 16 GB Mac (GPU budget 10.67 GB, 0.5 GB margin).
    assert fit_badge(7.0, 16, "gemma-4-12b-it-UD-Q4_K_XL.gguf") == "fits"     # 7.0 + 2.2 <= 10.17
    assert fit_badge(8.7, 16, "Qwen3.8-27B-UD-IQ2_S.gguf") == "too_big"      # 8.7 * 1.47 > 10.67
    assert fit_badge(28.0, 16) == "too_big"
    # Label parsing.
    assert _parse_label("gemma-4-12b-it-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert _parse_label("Qwen3.6-35B-A3B-Q8_0.gguf") == "Q8_0"
    assert _parse_label("model-BF16.gguf") == "BF16"
    assert _parse_label("model-IQ2_M.gguf") == "IQ2_M"
    print("self-test ok")
    # Live smoke test (best-effort; offline-safe).
    qs = fetch_quants("unsloth/gemma-4-12b-it-GGUF")
    print(f"fetched {len(qs)} quants")
    for q in qs[:5]:
        print(f"  {q.label:14} {q.size_gb:6.2f} GB  mmproj={q.is_mmproj}  {q.filename}")
