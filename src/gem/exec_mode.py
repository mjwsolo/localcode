from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .app import GemApp
from .config import AppConfig


def run_exec(config: AppConfig, prompt: str, cwd: Path, profile_name: str | None, model_name: str | None) -> int:
    console = Console()
    app = GemApp(config=config, cwd=cwd, profile_name=profile_name, model_name=model_name)
    try:
        response = app.ask(prompt, stream=False)
        console.print(response)
        return 0
    finally:
        app.close()
