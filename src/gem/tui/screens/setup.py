"""First-launch setup screen — downloads server and model."""
from __future__ import annotations

import platform
import subprocess

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

class SetupScreen(Screen):
    """Shows bootstrap progress during first launch."""

    DEFAULT_CSS = """
    SetupScreen {
        layout: vertical;
    }
    #setup-header {
        dock: top;
        height: 1;
        padding: 0 1;
        color: #5f87ff;
        background: $surface;
    }
    #setup-spacer {
        height: 1fr;
    }
    #setup-center {
        height: 1fr;
        align: center middle;
    }
    #setup-box {
        width: 52;
        height: auto;
        padding: 1 2;
        border: round #5f87ff;
    }
    #setup-status {
        width: 100%;
        color: $text-muted;
        margin: 1 0 0 0;
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
        from textual.containers import Vertical, Container
        yield Static("", id="setup-header")
        with Container(id="setup-center"):
            with Vertical(id="setup-box"):
                yield Static(self._render_steps(), id="setup-steps")
                yield Static("", id="setup-status")

    def on_resize(self) -> None:
        self._update_header()

    def _update_header(self) -> None:
        try:
            width = self.app.size.width or 80
        except Exception:
            width = 80
        usable = width - 2
        left = "🏠 LocalCode"
        left_cols = 14
        remaining = max(0, usable - left_cols)
        line = f"{left} {'─' * remaining}"
        self.query_one("#setup-header", Static).update(line)

    def _render_steps(self) -> str:
        lines = ["[bold]Setup[/]\n"]
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
        self._update_header()
        self.set_interval(0.1, self._tick)
        self.run_worker(self._run_setup, thread=True)

    def _tick(self) -> None:
        if self._failed_step >= 0:
            return  # stop animating on error
        self._spin_idx += 1
        try:
            # Update status text (countdown) and spinner together
            status = f"[dim]{self._status_text}[/]" if self._status_text else ""
            spin = self._spin_chars[self._spin_idx % len(self._spin_chars)]
            steps = self._render_steps()
            self.query_one("#setup-status", Static).update(status)
            self.query_one("#setup-steps", Static).update(steps)
            self.refresh()
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

        # ── Pre-flight: RAM check ──
        try:
            mem_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip())
            memory_gb = mem_bytes // (1024 ** 3)
            if memory_gb < 8:
                self.app.call_from_thread(lambda m=memory_gb: self._show_error(
                    f"LocalCode requires at least 8GB of RAM.\n"
                    f"Your Mac has {m}GB."
                ))
                return
        except Exception:
            pass

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

        # Set config for llama_cpp (GPU) or leave for Ollama fallback (decided later)
        changed = False
        if not config.runtime.llama_cpp_binary and binary_path:
            config.runtime.llama_cpp_binary = str(binary_path)
            changed = True
        # Default to llama_cpp — will be overridden to Ollama later if GPU unavailable
        if config.runtime.provider not in ("llama_cpp", "ollama"):
            config.runtime.provider = "llama_cpp"
            changed = True
        if config.runtime.provider == "llama_cpp" and "8081" not in config.runtime.base_url:
            config.runtime.base_url = "http://localhost:8081"
            changed = True
        if changed:
            save_config(config)

        # ── Step 1: Model ──
        self._current_step = 1
        self._status_text = "Checking model..."

        model_path = get_model_path()
        if not model_path:
            # Check disk space before downloading 10GB model
            try:
                import shutil
                free_gb = shutil.disk_usage(Path.home()).free / (1024 ** 3)
                if free_gb < 12:
                    self.app.call_from_thread(lambda f=free_gb: self._show_error(
                        f"Not enough disk space to download the model.\n"
                        f"Need ~12GB free, you have {f:.1f}GB.\n\n"
                        f"Free up space and try again."
                    ))
                    return
            except Exception:
                pass
            self._status_text = "Downloading model (~10GB)..."
            ok, result = download_model(
                on_progress=lambda msg: setattr(self, '_status_text', msg)
            )
            if not ok:
                err = str(result).replace("\n", " ").strip()[:80]
                self.app.call_from_thread(lambda e=err: self._show_error(
                    f"Model download failed:\n{e}\n\n"
                    f"Check your internet connection and try again."
                ))
                return
            model_path = result
            config.runtime.model = str(result)
            save_config(config)

        # Ensure Ollama has the model registered
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

        # Kill anything on port 8081 to free it
        try:
            r = subprocess.run(["lsof", "-ti", ":8081"], capture_output=True, text=True, timeout=3)
            pids = r.stdout.strip().split()
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
        except Exception:
            pass
        try:
            subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True, timeout=3)
        except Exception:
            pass
        time.sleep(1)

        # Actually launch llama-server — try GPU first, fall back to CPU on OOM
        from pathlib import Path
        import os
        from ...runtime import GemRuntimeGateway

        log_dir = Path.home() / ".local" / "share" / "localcode"
        log_dir.mkdir(parents=True, exist_ok=True)
        server_log = log_dir / "server.log"
        env = dict(os.environ)
        env["GGML_BACKEND_PATH"] = ""  # prevent system ggml from overriding TurboQuant

        # Ensure GPU is unlocked — installs persistent LaunchDaemon on first run
        from ...performance import ensure_gpu_unlock
        self._status_text = "Checking GPU..."
        gpu_ready = ensure_gpu_unlock()

        if not gpu_ready:
            try:
                mem_bytes = int(subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip())
                memory_gb = mem_bytes // (1024 ** 3)
            except Exception:
                memory_gb = 8

            if memory_gb < 16:
                # 8GB: run with CPU-only mode and smaller context
                self._status_text = "Configuring for 8GB (CPU mode)..."
                config.runtime.llama_cpp_gpu_layers = 0
                config.runtime.max_context_chars = 32768 * 4  # 32K tokens
                save_config(config)
                gpu_ready = True
            else:
                # 16GB: need GPU unlock
                # First try: native macOS password dialog (osascript)
                self._status_text = "Requesting GPU access..."
                from ...performance import _sudo_via_osascript, metal_gpu_available
                if _sudo_via_osascript("/usr/sbin/sysctl iogpu.wired_limit_mb=14336"):
                    if metal_gpu_available(memory_gb):
                        gpu_ready = True

                if not gpu_ready:
                    # osascript failed — exit TUI, explain, run sudo in terminal
                    def _exit_and_unlock():
                        self.app.exit(return_code=42)
                    self.app.call_from_thread(_exit_and_unlock)
                    return

        # Launch server (runs for ALL paths: 8GB CPU, 16GB GPU, 32GB+ GPU)
        if gpu_ready:
            gw = GemRuntimeGateway(config.runtime)
            from ...bootstrap import _turboquant_binary_path as _tbp
            binary = config.runtime.llama_cpp_binary or str(_tbp() or "")

            if not binary:
                self.app.call_from_thread(lambda: self._show_error("No server binary found. Reinstall localcode."))
                return
            if not model_path:
                self.app.call_from_thread(lambda: self._show_error("No model found. Delete ~/.gem/config.toml and retry."))
                return

            cmd = gw.llama_server_command(str(model_path))

            try:
                log_fh = open(server_log, "w")
                proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, start_new_session=True, env=env)
            except Exception as exc:
                self.app.call_from_thread(lambda e=str(exc): self._show_error(f"Failed to start: {e}"))
                return

            # Wait for server to become ready
            # 8GB CPU mode needs longer — model loads slower via mmap without GPU
            is_cpu_mode = config.runtime.llama_cpp_gpu_layers == 0
            wait_time = 180 if is_cpu_mode else 60
            server_ok = False
            for i in range(wait_time):
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
                if is_cpu_mode:
                    self._status_text = f"Loading model (CPU mode, this takes a while)... ({i+1}s)"
                else:
                    self._status_text = f"Loading model... ({i+1}s)"

            if not server_ok:
                # Server failed — fall back to Ollama instead of broken CPU mode
                err = ""
                try:
                    log_fh.close()
                    err = server_log.read_text()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                self._status_text = "GPU server failed — switching to Ollama..."
                time.sleep(1)
                config.runtime.provider = "ollama"
                config.runtime.base_url = "http://localhost:11434"
                if not config.runtime.model or config.runtime.model.endswith(".gguf"):
                    config.runtime.model = "gemma26b-iq3"
                config.runtime.laptop_26b_runtime_mode = "speed"
                from ...config import save_config
                save_config(config)

                # Check if Ollama is available as fallback
                try:
                    import httpx
                    r = httpx.get("http://localhost:11434/api/tags", timeout=3)
                    server_ok = r.status_code == 200
                except Exception:
                    pass
                if not server_ok:
                    self.app.call_from_thread(lambda e=err[-300:]: self._show_error(f"Server failed and Ollama not available:\n{e}"))
                    return

        # Done
        self._current_step = 3
        self._status_text = "Ready!"

        import asyncio
        await asyncio.sleep(0.5)
        self.app.call_from_thread(self._finish)

    def _finish(self) -> None:
        self.app.switch_screen("mode_picker")
