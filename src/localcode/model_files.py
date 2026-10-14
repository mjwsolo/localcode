"""Locate existing GGUF weights without copying or downloading them."""
from __future__ import annotations

import os
from pathlib import Path

_SKIP = {'.git', 'node_modules', '.venv', 'venv', '__pycache__', 'dist', 'build', 'voice'}


def search_roots(models_dir: Path) -> list[Path]:
    home = Path.home()
    roots = [models_dir]
    try:
        from .config import load_config
        configured = (load_config().runtime.model or '').strip()
        if configured and Path(configured).expanduser().is_file():
            roots.append(Path(configured).expanduser().parent)
    except (OSError, ValueError):
        pass
    roots.extend([
        home / '.local/share/localcode/models', home / 'models',
        Path(os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or
             str(Path(os.environ.get('HF_HOME', home / '.cache/huggingface')) / 'hub')),
        home / '.lmstudio/models', home / '.cache/lm-studio/models',
    ])
    return list(dict.fromkeys(path.expanduser().absolute() for path in roots))


def model_files(models_dir: Path) -> dict[str, Path]:
    """Configured directory wins duplicates; nested snapshots and symlinks work.

    Search only model directories, never the whole user home. Bound traversal
    and do not follow directory symlinks; file symlinks are ordinary weights.
    """
    found: dict[str, Path] = {}
    directories = 0
    for root in search_roots(models_dir):
        if not root.is_dir():
            continue
        for directory, children, files in os.walk(root, followlinks=False):
            directories += 1
            if directories > 5000:
                return found
            depth = len(Path(directory).relative_to(root).parts)
            children[:] = sorted(name for name in children if name not in _SKIP) if depth < 6 else []
            for name in sorted(files):
                if not name.lower().endswith('.gguf') or name.lower().startswith(('mmproj', 'mtp-', 'mtp_', 'dflash')):
                    continue
                path = Path(directory) / name
                if path.is_file():
                    found.setdefault(name, path)
    return found


def find_model(filename: str, models_dir: Path) -> Path | None:
    direct = models_dir / filename
    if direct.is_file():
        return direct
    return model_files(models_dir).get(filename)
