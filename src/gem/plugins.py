from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from .config import ensure_home_dirs


class PluginRegistry:
    def __init__(self) -> None:
        self.tool_builders: list[Callable[[], list[object]]] = []
        self.plugin_errors: list[str] = []

    def add_tool_builder(self, builder: Callable[[], list[object]]) -> None:
        self.tool_builders.append(builder)


def plugin_dirs(repo_root: Path) -> list[Path]:
    return [
        ensure_home_dirs() / "plugins",
        repo_root / ".gem" / "plugins",
    ]


def load_plugins(repo_root: Path, registry: PluginRegistry) -> list[str]:
    loaded: list[str] = []
    errors: list[str] = []
    for directory in plugin_dirs(repo_root):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(f"gem_plugin_{path.stem}", path)
                if spec is None or spec.loader is None:
                    errors.append(f"{path.stem}: invalid module spec")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                register = getattr(module, "register", None)
                if callable(register):
                    register(registry)
                    loaded.append(path.stem)
                else:
                    errors.append(f"{path.stem}: missing register(registry)")
            except Exception as exc:
                errors.append(f"{path.stem}: {exc}")
    registry.plugin_errors = errors
    return loaded
