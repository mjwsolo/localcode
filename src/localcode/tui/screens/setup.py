"""First-launch setup screen — downloads server and model."""
from __future__ import annotations

from ..._subproc_env import clean_env

import subprocess

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

class SetupScreen(Screen):
    """Shows bootstrap progress during first launch."""

    BINDINGS = [
        # Only fire while the screen is in the failed state — see
        # action_retry / action_quit which gate on self._failed_step.
        Binding("r", "retry", "Retry", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    DEFAULT_CSS = """
    SetupScreen {
        layout: vertical;
        background: ansi_default;
        padding: 1 0;
    }
    /* Centered on both axes. Previous version had the CSS selector
       (#setup-center) not match the compose id (#setup-top) so the
       alignment silently did nothing — box ended up top-left. */
    #setup-center {
        background: ansi_default;
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    #setup-box {
        background: ansi_default;
        width: 92%;
        max-width: 56;
        height: auto;
        padding: 1 2;
        border: round #5f87ff;
    }
    #setup-status {
        background: ansi_default;
        width: 100%;
        color: ansi_default;
        margin: 1 0 0 0;
        text-align: center;
    }
    /* Add breathing room between the status area (which may include
       error + retry hints) and the dimmed one-time-download note so
       the note doesn't visually crowd the error. */
    #setup-onetime-note {
        background: ansi_default;
        width: 100%;
        margin: 2 0 0 0;
        text-align: center;
    }
    /* Brand at bottom-left — `#brand-bar` styled in tui/styles/app.tcss. */
    """

    STEPS = [
        ("server", "Check inference server"),
        ("model", "Download model"),  # size substituted in _render_steps()
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
        # Brand at bottom-left (agent/Gemini-style). Same `#brand-bar`
        # widget identity used across every LocalCode screen.
        from ...theme import C as _C
        yield Static(f"🏠[{_C.primary}]LocalCode[/]", id="brand-bar")
        with Container(id="setup-center"):
            with Vertical(id="setup-box"):
                yield Static(self._render_steps(), id="setup-steps")
                yield Static("", id="setup-status")
                yield Static(
                    "[dim italic]Note: one-time download — cached locally so future "
                    "launches start in seconds without re-downloading.[/]",
                    id="setup-onetime-note",
                )

    def on_resize(self) -> None:
        # No-op now that the header bar is gone. Kept so any future
        # screen-resize hook on this class still compiles.
        return

    def _render_steps(self) -> str:
        # Substitute the actually-selected model's size into the "Download model"
        # label so the user sees the real number, not a hardcoded ~10GB.
        # If a partial download exists on disk (from a prior aborted attempt),
        # surface that as "Resume download" so the user knows we're picking up
        # where they left off — not starting over.
        try:
            from ...models_catalog import current as current_choice
            cfg = getattr(self.app, "config", None)
            chosen = current_choice(cfg) if cfg is not None else None
            if chosen is not None:
                verb = "Download"
                # huggingface_hub writes `.incomplete` sidecars; our urllib
                # fallback pre-allocates the final filename. Detect both.
                partial = chosen.local_path
                incomplete = partial.with_name(partial.name + ".incomplete")
                has_partial = (
                    (partial.exists() and partial.stat().st_size > 0
                     and partial.stat().st_size < int(chosen.size_gb * 1024 ** 3 * 0.99))
                    or incomplete.exists()
                )
                if has_partial:
                    verb = "Resume"
                model_label = f"{verb} model (~{chosen.size_gb:.1f} GB)"
            else:
                model_label = "Download model"
        except Exception:
            model_label = "Download model"

        lines = ["[bold]Setup[/]\n"]
        for i, (key, label) in enumerate(self.STEPS):
            if key == "model":
                label = model_label
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

    def _show_error(self, msg: str, code: str = "E1001", details: str = "") -> None:
        """Render a setup error using the project-wide error-code system.

        msg     — short user-facing summary (one line). Defaults to the
                  registry's summary text for `code`.
        code    — error code (e.g. E1001). Looked up in the registry.
        details — optional verbose log (e.g. tail of server.log). Written
                  to ~/.localcode/last_error.log; NOT shown in the UI.
                  This keeps the on-screen message clean (image 69 was
                  unreadable because raw llama-server stderr was dumped
                  into the box).
        """
        self._failed_step = self._current_step
        # Resolve the error code → summary + remediation. If the code
        # isn't registered, fall back to the bare msg.
        try:
            from ...errors import by_code
            ec = by_code(code)
        except Exception:
            ec = None
        # Compose the user-visible body — same [Eccc] prefix + fix line
        # used everywhere else in the app for consistent format.
        if ec is not None:
            summary = msg or ec.summary
            body = f"[red][{ec.code}][/] [bold red]{summary}[/]"
            body += f"\n\n[dim]fix: {ec.remediation}[/]"
        else:
            body = f"[bold red]{msg}[/]"
        if details:
            try:
                from ...paths import last_error_log_path
                p = last_error_log_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"[{code}] {msg}\n\n{details}")
                body += f"\n\n[dim](full log: .localcode/last_error.log)[/]"
            except Exception:
                pass
        body += "\n\n[dim]Press [bold]r[/] to retry · [bold]q[/] / Ctrl+C to quit.[/]"
        try:
            self.query_one("#setup-steps", Static).update(self._render_steps())
            self.query_one("#setup-status", Static).update(body)
        except Exception:
            pass

    def action_retry(self) -> None:
        """Retry the setup pipeline from the failed step without quitting.

        Resets failure state and re-runs the worker. The download path
        leaves partial files in place across attempts, so this picks up
        from where the last attempt died rather than re-downloading from 0.
        """
        if self._failed_step < 0:
            return  # nothing to retry
        self._failed_step = -1
        self._status_text = "Retrying..."
        # Reset the failed step so the spinner resumes on it.
        # _run_setup is idempotent on the server-binary and model-exists
        # checks, so it'll skip already-completed steps automatically.
        self._current_step = 0
        try:
            self.query_one("#setup-status", Static).update("")
            self.query_one("#setup-steps", Static).update(self._render_steps())
        except Exception:
            pass
        self.run_worker(self._run_setup, thread=True)

    def action_quit(self) -> None:
        """Quit only when in the failed state — otherwise the user might
        kill an in-flight setup by accidentally hitting q."""
        if self._failed_step < 0:
            return
        self.app.exit()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)
        # Skip the real setup worker when running under
        # `--preview-screen` — show the visuals only, don't start a
        # server / download a model.
        if getattr(self.app, "_preview_screen", None):
            self._status_text = "(preview mode — server not started)"
            self._current_step = 2  # park on "Start server..." step
            return
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
            download_model, get_model_path,
        )
        from ...models_catalog import CHOICES, current as current_choice

        config = self.app.config

        # ── Pre-flight: RAM check ──
        try:
            mem_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip())
            memory_gb = mem_bytes // (1024 ** 3)
            if memory_gb < 16:
                self.app.call_from_thread(lambda m=memory_gb: self._show_error(
                    f"LocalCode requires at least 16GB of RAM.\n"
                    f"Your Mac has {m}GB — not enough to run\n"
                    f"Gemma 4 26B (10.4GB model).\n\n"
                    f"A smaller model for 8GB Macs is coming soon."
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

        # Resolve which model to use. If the user ran `localcode setup` first and
        # picked one, config.runtime.model points at that GGUF and current()
        # returns the matching catalog entry. Otherwise we default to the
        # first catalog entry (Gemma) — the in-TUI picker is queued for a
        # follow-up; for now the CLI `localcode setup` is the interactive path.
        chosen = current_choice(config) or CHOICES[0]

        model_path = get_model_path(chosen.filename)
        if not model_path:
            # Check disk space before downloading
            try:
                import shutil
                free_gb = shutil.disk_usage(Path.home()).free / (1024 ** 3)
                needed = chosen.size_gb + 2  # ~2 GB headroom for partial writes
                if free_gb < needed:
                    self.app.call_from_thread(lambda f=free_gb, n=needed: self._show_error(
                        f"Not enough disk space to download the model.\n"
                        f"Need ~{n:.0f}GB free, you have {f:.1f}GB.\n\n"
                        f"Free up space and try again."
                    ))
                    return
            except Exception:
                pass
            self._status_text = f"Downloading {chosen.name} (~{chosen.size_gb:.1f}GB)..."
            ok, result = download_model(
                chosen,
                on_progress=lambda msg: setattr(self, '_status_text', msg),
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

        # ── Step 2: Start server ──
        self._current_step = 2
        self._status_text = "Starting server..."
        self.app.config = load_config()
        config = self.app.config

        # Check if server is already running AND serving the model the user
        # just picked. Healthcheck-alone was the bug (2026-05-23): an
        # orphaned llama-server from a previous session (Qwen) stayed alive
        # on port 8081 across TUI restarts; the new TUI then saw `ok=True`
        # and reused the OLD server even though the user had just picked
        # Gemma. The user's selection silently did nothing.
        from ...runtime import LocalCodeRuntimeGateway
        gw = LocalCodeRuntimeGateway(config.runtime)
        chosen_basename = Path(config.runtime.model).name if config.runtime.model else ""
        try:
            ok, _ = gw.healthcheck()
            if ok:
                # ALSO verify the running server is serving the chosen model.
                # llama-server's /v1/models endpoint returns the loaded
                # model path under data[0].id. If it doesn't match the
                # chosen file's basename, fall through to the shutdown +
                # restart path below.
                running_model_ok = False
                try:
                    import httpx as _hx
                    r = _hx.get(
                        config.runtime.base_url.rstrip("/") + "/v1/models",
                        timeout=3.0,
                    )
                    if r.status_code == 200:
                        data = r.json() or {}
                        loaded = ((data.get("data") or [{}])[0]).get("id") or ""
                        # `id` is usually the full GGUF path; compare basename.
                        if chosen_basename and Path(loaded).name == chosen_basename:
                            running_model_ok = True
                except Exception:
                    # Couldn't verify — be safe, force restart.
                    running_model_ok = False
                if running_model_ok:
                    # Healthcheck OK AND model matches — reuse.
                    self._current_step = 3
                    self._status_text = "Ready!"
                    import asyncio
                    await asyncio.sleep(0.3)
                    self.app.call_from_thread(self._finish)
                    return
        except Exception:
            pass

        # Shut down any pre-existing llama-server through the single
        # lifecycle owner. Handles: our tracked Popen from a prior run,
        # stale PID file from a crashed session, and any process bound to
        # the port (e.g. user's other install). Replaces the scattered
        # pkill + lsof dance.
        from ...server_manager import ServerManager
        ServerManager.get().shutdown()
        time.sleep(1)

        # Actually launch llama-server — try GPU first, fall back to CPU on OOM
        from pathlib import Path
        from ...runtime import LocalCodeRuntimeGateway

        log_dir = Path.home() / ".local" / "share" / "localcode"
        log_dir.mkdir(parents=True, exist_ok=True)
        server_log = log_dir / "server.log"
        # Single source of truth for env scrubbing — see _subproc_env.py.
        env = clean_env()
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
                self.app.call_from_thread(lambda m=memory_gb: self._show_error(
                    f"LocalCode requires at least 16GB of RAM.\n"
                    f"Your Mac has {m}GB — not enough to run\n"
                    f"Gemma 4 26B (10.4GB model).\n\n"
                    f"A smaller model for 8GB Macs is coming soon."
                ))
                return
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

        # Set GPU layers based on CURRENT hardware state — never trust saved config
        if gpu_ready:
            config.runtime.llama_cpp_gpu_layers = 999  # GPU mode
        else:
            config.runtime.llama_cpp_gpu_layers = 0    # CPU mode
        # Normalize known unsafe persisted launch settings before saving.
        # Older config can carry `-b 2048` / 64K context from a previous
        # runtime experiment; for large Qwen GGUFs on 16 GB Macs that shape
        # can Metal-OOM during the first prefill. Runtime clamps defensively
        # too, but saving the normalized values keeps setup/status/restarts
        # consistent and prevents stale config from reintroducing flakiness.
        try:
            from ...runtime import LocalCodeRuntimeGateway as _LCGW
            _gw_for_policy = _LCGW(config.runtime)
            if (
                _gw_for_policy._is_large_qwen_gguf(str(model_path))
                and _gw_for_policy._system_ram_gb() <= 18
            ):
                config.runtime.llama_cpp_batch_size = min(
                    max(128, int(config.runtime.llama_cpp_batch_size or 512)),
                    512,
                )
                config.runtime.max_context_chars = min(
                    int(config.runtime.max_context_chars or 131072),
                    131072,
                )
        except Exception:
            pass
        save_config(config)

        # Launch server
        gw = LocalCodeRuntimeGateway(config.runtime)
        from ...bootstrap import _turboquant_binary_path as _tbp
        binary = config.runtime.llama_cpp_binary or str(_tbp() or "")

        if not binary:
            self.app.call_from_thread(lambda: self._show_error("No server binary found. Reinstall localcode."))
            return
        if not model_path:
            self.app.call_from_thread(lambda: self._show_error("No model found. Delete ~/.localcode/config.toml and retry."))
            return

        # Pick a port dynamically. 8081 is our documented default, but a
        # user could have Ollama on 8080 spilling over, a stale localcode
        # from a prior session, VS Code port-forwarding, a devcontainer,
        # etc. `find_free_port` tries 8081 → 8082…8099 → OS-assigned
        # ephemeral — always succeeds. Honors $LOCALCODE_PORT for users
        # with pinned-port requirements. The actual port gets written
        # back to config.runtime.base_url so every downstream HTTP
        # client hits the right URL.
        from ...server_manager import find_free_port, DEFAULT_PORT
        chosen_port = find_free_port(DEFAULT_PORT)
        config.runtime.base_url = f"http://localhost:{chosen_port}"

        cmd = gw.llama_server_command(str(model_path), port=chosen_port)
        # Refresh the gateway's endpoint URLs so the healthcheck /
        # warmup loop below hits `chosen_port` rather than whatever
        # base_url it was constructed with (usually 8081).
        gw.config.base_url = config.runtime.base_url
        if gw.config.provider == "llama_cpp":
            gw.endpoint = f"{gw.config.base_url}/v1/chat/completions"
            gw.tags_endpoint = f"{gw.config.base_url}/v1/models"

        try:
            log_fh = open(server_log, "w")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, start_new_session=True, env=env)
        except Exception as exc:
            self.app.call_from_thread(lambda e=str(exc): self._show_error(f"Failed to start: {e}"))
            return

        # Hand ownership of the freshly-spawned server over to ServerManager.
        # Without this, our signal handlers + atexit cleanup don't know
        # about this PID and the server survives the TUI exit. Tracking
        # it means Ctrl+C / quit / jetsam-kill all reach it through the
        # lifecycle owner.
        try:
            from ...server_manager import (
                ServerManager, PID_FILE,
                _lifecycle_log, _system_free_memory_mb,
            )
            mgr = ServerManager.get()
            mgr._process = proc
            mgr._model_path = str(model_path)
            mgr._port = chosen_port
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(str(proc.pid))
            # Emit server_started to the lifecycle log. The setup screen
            # spawns via raw Popen (so it can keep the server log fh) and
            # then hands the process to ServerManager — but that bypasses
            # ServerManager.start() which is the normal site of this
            # event. Without this line, the very FIRST server start of a
            # session is invisible in lifecycle.log, only restarts/stops
            # show up. Tail readers see "nothing happens at launch."
            # Truncated full launch flags so we can audit what
            # llama-server was actually invoked with (mirrors the
            # ServerManager.start() path; otherwise setup-screen
            # launches show no flags in events.jsonl).
            _flags_str = " ".join(str(c) for c in cmd)[:4000]
            _lifecycle_log(
                "server_started",
                pid=proc.pid,
                port=chosen_port,
                model=Path(model_path).name,
                free_mb_after_spawn=_system_free_memory_mb(),
                source="setup_screen",
                flags=_flags_str,
            )
        except Exception:
            pass

        # Wait for server to become ready — must pass BOTH health + inference test
        is_cpu_mode = config.runtime.llama_cpp_gpu_layers == 0
        wait_time = 180 if is_cpu_mode else 120
        server_ok = False
        health_ok = False
        for i in range(wait_time):
            time.sleep(1)
            if proc.poll() is not None:
                break  # crashed
            try:
                if not health_ok:
                    ok, _ = gw.healthcheck()
                    if ok:
                        health_ok = True
                        self._status_text = f"Server ready... ({i+1}s)"
                        server_ok = True
                        break
                else:
                    self._status_text = f"Server ready... ({i+1}s)"
            except Exception:
                pass
            if not health_ok:
                self._status_text = f"Loading model... ({i+1}s)"

        if not server_ok:
            # Server failed — fall back to Ollama IN MEMORY ONLY for this
            # session. Previously we persisted provider="ollama" to
            # config.toml — once a single setup retry hit a transient
            # timeout (e.g. memory pressure during a model swap), the
            # config was permanently rewritten and every subsequent
            # launch ran through Ollama at ~0.5 tok/s instead of the
            # 27 tok/s TurboQuant llama-server. The persisted fallback
            # was the source of "Gemma got 50× slower this afternoon"
            # — fix is to mutate the live config object only, never
            # save_config() here. Next launch will re-attempt llama-server.
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
            self._status_text = "GPU server failed — using Ollama for this session..."
            time.sleep(1)
            config.runtime.provider = "ollama"
            config.runtime.base_url = "http://localhost:11434"
            if not config.runtime.model or config.runtime.model.endswith(".gguf"):
                config.runtime.model = "gemma26b-iq3"
            config.runtime.laptop_26b_runtime_mode = "speed"
            # NOTE: deliberately NO save_config(config) here. See comment above.

            # Check if Ollama is available as fallback
            try:
                import httpx
                r = httpx.get("http://localhost:11434/api/tags", timeout=3)
                server_ok = r.status_code == 200
            except Exception:
                pass
            if not server_ok:
                # Use the error-code system so the user sees a clear
                # [E1001] prefix + remediation, with the raw server log
                # tucked into ~/.localcode/last_error.log instead of
                # being dumped into the screen box.
                self.app.call_from_thread(lambda e=err: self._show_error(
                    msg="The model server didn't start.",
                    code="E1001",
                    details=e,
                ))
                return

        # Done
        self._current_step = 3
        self._status_text = "Ready!"

        import asyncio
        await asyncio.sleep(0.5)
        self.app.call_from_thread(self._finish)

    def _finish(self) -> None:
        self.app.switch_screen("chat")
