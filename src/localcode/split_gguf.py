"""Split (sharded) GGUF helpers.

Hugging Face caps a file at 50 GB, so big quants ship as shards named
``<stem>-00001-of-00003.gguf`` (sometimes in a subfolder). llama-server loads
the whole model when given the FIRST shard with the others beside it.

LocalCode stores shards flat in the models dir with their original names. The
quant is identified by its first shard's filename, and its alias (the name shown
in the UI and passed to ``--alias``) is the stem, e.g. ``flint-BF16``.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$", re.IGNORECASE)


def parse_shard(name: str) -> tuple[str, int, int] | None:
    """(stem, index, total) for a shard filename (directories ignored), else None."""
    m = SHARD_RE.match(name.rsplit("/", 1)[-1])
    if not m:
        return None
    return m.group("stem"), int(m.group("idx")), int(m.group("n"))


def is_shard(name: str) -> bool:
    return parse_shard(name) is not None


def is_first_shard(name: str) -> bool:
    p = parse_shard(name)
    return p is not None and p[1] == 1


def is_extra_shard(name: str) -> bool:
    """A shard that is not the first one (never a loadable model on its own)."""
    p = parse_shard(name)
    return p is not None and p[1] != 1


def shard_names(first: str) -> list[str]:
    """Every shard filename of the split whose first shard is ``first``."""
    p = parse_shard(first)
    if p is None:
        return [first.rsplit("/", 1)[-1]]
    stem, _, n = p
    return [f"{stem}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def display_alias(filename: str) -> str:
    """Alias for a model file: the name without ``.gguf``, and for a split
    without the ``-NNNNN-of-NNNNN`` part (``flint-BF16``)."""
    base = filename.rsplit("/", 1)[-1]
    p = parse_shard(base)
    if p is not None:
        return p[0]
    return base.removesuffix(".gguf")


def shard_paths(first: Path) -> list[Path]:
    """Paths of all shards, beside ``first``. A plain file gives ``[first]``."""
    if not is_shard(first.name):
        return [first]
    return [first.with_name(n) for n in shard_names(first.name)]


def total_bytes(first: Path) -> int:
    """Combined size of the model's files on disk (a plain file is its own size)."""
    total = 0
    for p in shard_paths(first):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def split_complete(first: Path, sizes: dict[str, int] | None = None) -> bool:
    """True when the model is usable. A plain file just has to exist. A split
    needs every shard beside the first, each with its expected size when
    ``sizes`` (shard filename -> bytes) is known, and non-empty otherwise."""
    if not is_shard(first.name):
        return first.is_file()
    for p in shard_paths(first):
        try:
            size = p.stat().st_size
        except OSError:
            return False
        want = (sizes or {}).get(p.name)
        if want:
            if size != want:
                return False
        elif size <= 0:
            return False
    return True


def resolve_model_path(models_dir: Path, alias: str) -> Path:
    """Path to hand llama-server for ``alias``: ``<alias>.gguf`` when that file
    exists, else the first shard of the split called ``alias``, else the plain
    path (which then simply does not exist)."""
    plain = models_dir / f"{alias}.gguf"
    if plain.is_file():
        return plain
    try:
        for p in sorted(models_dir.glob(f"{glob.escape(alias)}-00001-of-*.gguf")):
            parsed = parse_shard(p.name)
            if parsed and parsed[0] == alias and p.is_file():
                return p
    except OSError:
        pass
    return plain


def link_split(existing_first: Path, models_dir: Path) -> None:
    """Symlink every shard found beside ``existing_first`` into ``models_dir``."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for src in shard_paths(existing_first.resolve()):
        dst = models_dir / src.name
        if src.is_file() and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)


def remove_split(first: Path) -> list[str]:
    """Delete every shard of a split (or the single file). Returns names removed."""
    removed = []
    for p in shard_paths(first):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed.append(p.name)
        except OSError:
            continue
    return removed
