"""First-launch setup screen — downloads server and model."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

_HEADER = "──────────────── 🏠 LocalCode ────────────────"


class SetupScreen(Screen):
    """Shows bootstrap progress during first launch."""

    DEFAULT_CSS = """
    SetupScreen {
        align: center middle;
    }
    #setup-wrap {
        width: 52;
        height: auto;
    }
    #setup-header {
        width: 100%;
        text-align: center;
        color: $primary;
        margin-bottom: 1;
    }
    #setup-box {
        width: 100%;
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    #setup-status {
        width: 100%;
        color: $text-muted;
        margin-top: 1;
        text-align: center;
    }
    """

    STEPS = [
        ("server", "Check inference server"),
        ("model", "Download model (~10GB)"),
        ("start", "Start server"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._current_step = 0
        self._total_steps = len(self.STEPS)
        self._spin_idx = 0
        self._spin_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        with Vertical(id="setup-wrap"):
            yield Static(_HEADER, id="setup-header")
            with Vertical(id="setup-box"):
                yield Static(self._render_steps(), id="setup-steps")
                yield Static("", id="setup-status")

    def _render_steps(self) -> str:
        lines = ["[bold]First-time setup[/]\n"]
        for i, (key, label) in enumerate(self.STEPS):
            if i < self._current_step:
                lines.append(f"  [green]✓[/] {label}")
            elif i == self._current_step:
                spin = self._spin_chars[self._spin_idx % len(self._spin_chars)]
                lines.append(f"  [bold yellow]{spin}[/] {label}...")
            else:
                lines.append(f"  [dim]○[/] {label}")
        return "\n".join(lines)

    def _update(self, step: int, status: str = "") -> None:
        self._current_step = step
        try:
            self.query_one("#setup-steps", Static).update(self._render_steps())
            if status:
                self.query_one("#setup-status", Static).update(f"[dim]{status}[/]")
        except Exception:
            pass

    def _show_error(self, msg: str) -> None:
        try:
            self.query_one("#setup-status", Static).update(
                f"[red]{msg}[/]\n\n[dim]Press Ctrl+C to quit.\nRetry with 'localcode'.[/]"
            )
        except Exception:
            pass

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)
        self.run_worker(self._run_setup, thread=True)

    def _tick(self) -> None:
        self._spin_idx += 1
        try:
            self.query_one("#setup-steps", Static).update(self._render_steps())
        except Exception:
            pass

    async def _run_setup(self) -> None:
        import time
        from ...config import load_config, save_config
        from ...bootstrap import (
            _turboquant_binary_path, _find_turboquant_source,
            build_turboquant, download_turboquant_binary,
            is_ollama_installed, install_ollama, pull_model,
        )

        config = self.app.gem_config

        # ── Step 0: Server binary ──
        self.app.call_from_thread(lambda: self._update(0, "Checking server..."))

        binary_path = _turboquant_binary_path()
        if not binary_path and config.runtime.llama_cpp_binary:
            from pathlib import Path
            p = Path(config.runtime.llama_cpp_binary)
            if p.exists():
                binary_path = p

        if not binary_path:
            self.app.call_from_thread(lambda: self._update(0, "Downloading server..."))
            if _find_turboquant_source():
                ok, result = build_turboquant()
            else:
                ok, result = download_turboquant_binary(on_progress=lambda _: None)
            if not ok:
                self.app.call_from_thread(lambda: self._show_error("Server download failed."))
                return
            binary_path = result
            config.runtime.llama_cpp_binary = str(binary_path)
            save_config(config)

        # ── Step 1: Model via Ollama ──
        self.app.call_from_thread(lambda: self._update(1, "Checking Ollama..."))
        if not is_ollama_installed():
            from rich.console import Console
            ok, _ = install_ollama(Console(quiet=True))
            if not ok:
                self.app.call_from_thread(lambda: self._show_error("Install Ollama: ollama.com/download"))
                return

        import subprocess
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(2)

        resolved_model = config.runtime.model or "gemma26b-iq3"

        # Check if model already exists locally before pulling
        try:
            check = subprocess.run(["ollama", "show", resolved_model],
                                   capture_output=True, text=True, timeout=10)
            model_exists = check.returncode == 0
        except Exception:
            model_exists = False

        if not model_exists:
            self.app.call_from_thread(lambda: self._update(1, f"Downloading {resolved_model}..."))
            ok, result = pull_model(resolved_model, on_progress=lambda _: None)
            if not ok:
                err = str(result).replace("\n", " ").strip()[:60]
                self.app.call_from_thread(lambda e=err: self._show_error(e))
                return

        # ── Step 2: Start server ──
        self.app.call_from_thread(lambda: self._update(2, "Starting server..."))
        self.app.gem_config = load_config()

        # Done
        self.app.call_from_thread(lambda: self._update(3, "Ready!"))

        import asyncio
        await asyncio.sleep(0.5)
        self.app.call_from_thread(self._finish)

    def _finish(self) -> None:
        self.app.switch_screen("mode_picker")
