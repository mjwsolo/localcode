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
        self._failed_step = -1  # which step failed (-1 = none)
        self._status_text = ""

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
            if i == self._failed_step:
                lines.append(f"  [red]✗[/] {label}")
            elif i < self._current_step:
                lines.append(f"  [green]✓[/] {label}")
            elif i == self._current_step and self._failed_step == -1:
                spin = self._spin_chars[self._spin_idx % len(self._spin_chars)]
                lines.append(f"  [bold yellow]{spin}[/] {label}...")
            else:
                lines.append(f"  [dim]○[/] {label}")
        return "\n".join(lines)

    def _update(self, step: int, status: str = "") -> None:
        self._current_step = step
        if status:
            self._status_text = status

    def _show_error(self, msg: str) -> None:
        self._failed_step = self._current_step
        try:
            self.query_one("#setup-steps", Static).update(self._render_steps())
            self.query_one("#setup-status", Static).update(
                f"[red]{msg}[/]\n\n[dim]Press Ctrl+C to quit.\nRetry with 'localcode'.[/]"
            )
        except Exception:
            pass

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)
        self.run_worker(self._run_setup, thread=True)

    def _tick(self) -> None:
        if self._failed_step >= 0:
            return  # stop animating on error
        self._spin_idx += 1
        try:
            self.query_one("#setup-steps", Static).update(self._render_steps())
            if hasattr(self, '_status_text') and self._status_text:
                self.query_one("#setup-status", Static).update(f"[dim]{self._status_text}[/]")
        except Exception:
            pass

    async def _run_setup(self) -> None:
        import time
        from ...config import load_config, save_config
        from ...bootstrap import (
            _turboquant_binary_path, _find_turboquant_source,
            build_turboquant, download_turboquant_binary,
            is_ollama_installed, install_ollama,
            download_model, get_model_path, create_ollama_model,
            _OLLAMA_MODEL_TAG,
        )

        config = self.app.gem_config

        # ── Step 0: Server binary ──
        self._current_step = 0
        self._status_text = "Checking server..."

        binary_path = _turboquant_binary_path()
        if not binary_path and config.runtime.llama_cpp_binary:
            from pathlib import Path
            p = Path(config.runtime.llama_cpp_binary)
            if p.exists():
                binary_path = p

        if not binary_path:
            self._status_text = "Downloading server..."
            if _find_turboquant_source():
                ok, result = build_turboquant()
            else:
                ok, result = download_turboquant_binary(on_progress=lambda _: None)
            if not ok:
                self.app.call_from_thread(lambda: self._show_error("Server download failed."))
                return
            binary_path = result
            config.runtime.llama_cpp_binary = str(binary_path)

        # Always ensure config is set for llama_cpp after we have a binary
        changed = False
        if config.runtime.provider != "llama_cpp":
            config.runtime.provider = "llama_cpp"
            changed = True
        if "8081" not in config.runtime.base_url:
            config.runtime.base_url = "http://localhost:8081"
            changed = True
        if not config.runtime.llama_cpp_binary and binary_path:
            config.runtime.llama_cpp_binary = str(binary_path)
            changed = True
        if changed:
            save_config(config)

        # ── Step 1: Model ──
        self._current_step = 1
        self._status_text = "Checking model..."

        model_path = get_model_path()
        if not model_path:
            self._status_text = "Downloading model (~10GB)..."
            ok, result = download_model(
                on_progress=lambda msg: setattr(self, '_status_text', msg)
            )
            if not ok:
                err = str(result).replace("\n", " ").strip()[:60]
                self.app.call_from_thread(lambda e=err: self._show_error(e))
                return
            model_path = result
            config.runtime.model = str(result)
            save_config(config)

        # Ensure Ollama has the model registered
        import subprocess
        if is_ollama_installed():
            try:
                check = subprocess.run(["ollama", "show", _OLLAMA_MODEL_TAG],
                                       capture_output=True, text=True, timeout=10)
                if check.returncode != 0:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    time.sleep(2)
                    self._status_text = "Registering model..."
                    ok, result = create_ollama_model(str(model_path))
                    if not ok:
                        err = str(result).replace("\n", " ").strip()[:60]
                        self.app.call_from_thread(lambda e=err: self._show_error(e))
                        return
            except Exception:
                pass

        # ── Step 2: Start server ──
        self._current_step = 2
        self._status_text = "Starting server..."
        self.app.gem_config = load_config()
        config = self.app.gem_config

        # Check if server is already running AND can actually serve requests
        from ...runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(config.runtime)
        try:
            ok, _ = gw.healthcheck()
            if ok:
                # Quick inference test to make sure it's not a zombie
                import httpx
                r = httpx.post(
                    f"{config.runtime.base_url.rstrip('/')}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=15,
                )
                if r.status_code == 200:
                    # Server is genuinely working — skip to done
                    self._current_step = 3
                    self._status_text = "Ready!"
                    import asyncio
                    await asyncio.sleep(0.3)
                    self.app.call_from_thread(self._finish)
                    return
        except Exception:
            pass

        # Kill any existing llama-server to free the port
        try:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True, timeout=3)
            time.sleep(1)
        except Exception:
            pass

        # Actually launch llama-server — try GPU first, fall back to CPU on OOM
        from pathlib import Path
        import os
        from ...runtime import GemRuntimeGateway

        log_dir = Path.home() / ".local" / "share" / "localcode"
        log_dir.mkdir(parents=True, exist_ok=True)
        server_log = log_dir / "server.log"
        env = dict(os.environ)
        env["GGML_BACKEND_PATH"] = ""  # prevent system ggml from overriding TurboQuant

        gw = GemRuntimeGateway(config.runtime)
        # Use bundled binary if config doesn't have one
        from ...bootstrap import _turboquant_binary_path as _tbp
        binary = config.runtime.llama_cpp_binary or str(_tbp() or "")

        if not binary:
            self.app.call_from_thread(lambda: self._show_error("No server binary found. Reinstall localcode."))
            return
        if not model_path:
            self.app.call_from_thread(lambda: self._show_error("No model found. Delete ~/.gem/config.toml and retry."))
            return

        if binary and model_path:
            # Try up to 2 attempts: first with current mode, then CPU-only fallback
            for attempt in range(2):
                cmd = gw.llama_server_command(str(model_path))

                try:
                    log_fh = open(server_log, "w")
                    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, start_new_session=True, env=env)
                except Exception as exc:
                    self.app.call_from_thread(lambda e=str(exc): self._show_error(f"Failed to start: {e}"))
                    return

                # Wait for server to become ready (up to 60s)
                server_ok = False
                for i in range(60):
                    time.sleep(1)
                    if proc.poll() is not None:
                        break  # crashed
                    try:
                        ok, _ = gw.healthcheck()
                        if ok:
                            server_ok = True
                            break
                    except Exception:
                        pass
                    self._status_text = f"Loading model... ({i+1}s)"

                if server_ok:
                    break

                # Server failed — check if it's an OOM/memory issue
                err = ""
                try:
                    log_fh.close()
                    err = server_log.read_text()
                except Exception:
                    pass

                is_oom = "working set size" in err or "out of memory" in err.lower() or "allocated size" in err

                if is_oom and attempt == 0:
                    # Kill the crashed/stuck server and retry with CPU-only
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    self._status_text = "GPU OOM — retrying CPU-only..."
                    time.sleep(1)
                    # Force CPU-only mode
                    config.runtime.laptop_26b_runtime_mode = "speed"
                    config.runtime.llama_cpp_gpu_layers = 0
                    from ...config import save_config
                    save_config(config)
                    gw = GemRuntimeGateway(config.runtime)
                    continue

                # Not OOM or second attempt also failed
                self.app.call_from_thread(lambda e=err[-300:]: self._show_error(f"Server failed:\n{e}"))
                return

        # Done
        self._current_step = 3
        self._status_text = "Ready!"

        import asyncio
        await asyncio.sleep(0.5)
        self.app.call_from_thread(self._finish)

    def _finish(self) -> None:
        self.app.switch_screen("mode_picker")
