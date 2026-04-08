from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import logging
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

from .approvals import ApprovalItem, ApprovalQueue
from .compact import compact_messages
from .composer import compose_messages
from .browser import browser_status, ensure_browser_mcp, find_browser_tool
from .config import AppConfig, ensure_home_dirs, save_config
from .context import build_context_block, find_repo_root, git_diff, git_status, list_repo_files, read_file
from .agent_background import launch_background_agent
from .cartridge import build_repo_cartridge
from .indexer import build_index, load_index, search_index
from .jobs import launch_background_job, list_jobs, read_job_log
from .mcp import load_mcp_configs
from .models import GEMMA_PROFILES, get_runtime_model, infer_profile_from_model, resolve_profile
from .onboarding import onboarding_panel
from .patching import apply_diff, build_diff, extract_last_diff_block, parse_diff
from .planner import build_plan_note
from .planner_hints import PlannerHint, PlannerHintState, parse_planner_hint
from .permissions import PermissionStore
from .prompts import build_system_prompt, build_task_appendix
from .runtime import GemRuntimeGateway, RuntimeErrorWithContext
from .session import SessionStore
from .shell import run_shell
from .audio_input import (
    AudioData as AudioInputData,
    audio_to_text_fallback,
    build_audio_message_hf,
    detect_audio_paths_in_text,
    is_audio_file,
    load_audio,
)
from .images import (
    ImageData,
    build_image_message,
    clipboard_has_image,
    detect_image_paths_in_text,
    load_image_from_path,
    read_clipboard_image,
    take_screenshot,
)
from .live_display import GemLiveDisplay
from .tool_router import route_tools, expand_tools_for_retry
from .cache import BackgroundIndexer, SpeculativeExecutor, ToolResultCache
from .network import is_online
from .watcher import FileWatcher, ProjectWatcher
from .keybindings import get_editing_mode, load_keybindings
from .plan_mode import Plan, PlanStep, parse_plan_from_response
from .permissions_v2 import PermissionManager
from .tasks import TaskStore
from .display import ThinkingIndicator, ToolCallDisplay, ResponseDisplay, SessionStats, DiffPreview, ContextBudgetDisplay
from .output import OutputManager
from .project_context import load_project_context, has_project_context
from .skills import list_skills, resolve_referenced_skills
from .toolkit import GemToolkit
from .embeddings import EmbeddingSearch
from .history import HistoryDB
from .traces import SessionLogger
from .ui_art import (
    GEM_BANNER,
    center_ascii_block,
    format_banner,
    format_tool_call,
    spinner_frame,
    thinking_frame,
    tool_icon,
    agent_progress_bar,
)
from .verification import build_verification_plan, guess_verify_command, run_verification
from .voice import speak_text, transcribe_audio, voice_status
from .auto_compact import compact_if_needed
from .autonomy import AutonomyLevel, apply_autonomy_to_permissions, format_autonomy_status, get_policy
from .hooks import HookRunner
from .notify import TaskNotifier, notify_if_slow
from .snapshots import SnapshotStore, create_snapshot, restore_snapshot
from .turn_diff import TurnDiffTracker, print_turn_diff
from .performance import detect_machine_profile, benchmark_report, apply_preset, should_promote_legacy_default_to_laptop_26b

import os
import time


_SLASH_COMMANDS = [
    "/help", "/status", "/model", "/files", "/find", "/index",
    "/read", "/add", "/drop", "/context", "/diff", "/apply",
    "/shell", "/bg", "/jobs", "/log", "/verify",
    "/agentbg",
    "/tools", "/skills", "/mcp", "/permissions",
    "/search", "/history", "/browser", "/voice",
    "/thinking", "/timeline",
    "/audio", "/image", "/paste", "/screenshot",
    "/undo", "/changes",
    "/autonomy", "/snapshot",
    "/clear", "/quit",
]

# Commands that take a file path as argument
_PATH_COMMANDS = {"/read", "/image", "/audio", "/add", "/drop", "/shell"}


class GemCompleter(Completer):
    """Completes /commands on Tab, and file paths for path-taking commands."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Complete /commands
        if text.startswith("/"):
            # Just the command itself (no space yet)
            if " " not in text:
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text))
                return
            # After the command, complete file paths for path commands
            cmd, _, arg = text.partition(" ")
            if cmd in _PATH_COMMANDS:
                yield from self._complete_path(arg)
                return
        # No completions for regular chat input
        return

    def _complete_path(self, partial: str):
        """Complete file paths relative to repo root."""
        try:
            if partial:
                base = self.repo_root / partial
                if base.is_dir():
                    parent = base
                    prefix = partial
                else:
                    parent = base.parent
                    prefix = partial
            else:
                parent = self.repo_root
                prefix = ""

            if not parent.is_dir():
                return

            search = base.name if partial and not (self.repo_root / partial).is_dir() else ""
            scan_dir = parent if partial and not (self.repo_root / partial).is_dir() else (self.repo_root / partial if partial else self.repo_root)

            if not scan_dir.is_dir():
                scan_dir = scan_dir.parent

            for entry in sorted(scan_dir.iterdir()):
                if entry.name.startswith("."):
                    continue
                rel = str(entry.relative_to(self.repo_root))
                if rel.startswith(partial):
                    suffix = "/" if entry.is_dir() else ""
                    yield Completion(rel + suffix, start_position=-len(partial))
        except Exception:
            return


class GemApp:
    def __init__(
        self,
        config: AppConfig,
        session_id: str | None = None,
        cwd: Path | None = None,
        profile_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.console = Console()
        self.log = logging.getLogger("gem")
        self.config = config
        machine = detect_machine_profile()
        if should_promote_legacy_default_to_laptop_26b(self.config, machine):
            _, preset = benchmark_report(self.config, self.config.runtime.mode)
            apply_preset(self.config, preset, model=self.config.runtime.model)
        self.repo_root = find_repo_root(cwd or Path.cwd())
        self.store = SessionStore()
        if session_id:
            self.session = self.store.load(session_id)
        else:
            self.session = self.store.create(
                self.repo_root,
                profile=profile_name or config.runtime.profile,
                model=model_name or config.runtime.model,
            )
        if profile_name:
            self.session.profile = profile_name
        if model_name is not None:
            self.session.model = model_name
        self.profile = resolve_profile(self.session.profile, self.session.model)
        self.runtime_model = get_runtime_model(self.profile, self.session.model)
        self.session.profile = self.profile.key
        self.session.model = self.runtime_model
        self.config.runtime.profile = self.profile.key
        self.config.runtime.model = self.runtime_model
        self.engine = GemRuntimeGateway(config.runtime)
        self.toolkit = GemToolkit(self.repo_root, config, app=self)
        self.permissions = PermissionStore(self.repo_root)
        self.store.save(self.session)
        history_path = ensure_home_dirs() / "prompt_history.txt"
        from prompt_toolkit.styles import Style
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        kb = KeyBindings()

        @kb.add("c-v")
        def _paste_image(event):
            """Ctrl+V — paste image from system clipboard."""
            if clipboard_has_image():
                img = read_clipboard_image()
                if img and not any(e.base64_data == img.base64_data for e in self._pending_images):
                    self._pending_images.append(img)
                    import sys as _s
                    _s.stdout.write(f"\n\033[2m  image attached ({img.size_kb}KB)\033[0m\n")
                    _s.stdout.flush()
            else:
                # No image — do normal text paste from system clipboard
                import subprocess as _sp
                try:
                    text = _sp.run(["pbpaste"], capture_output=True, text=True, timeout=2).stdout
                    if text:
                        event.current_buffer.insert_text(text)
                except Exception:
                    pass

        @kb.add(Keys.BracketedPaste)
        def _bracketed_paste(event):
            """Handle Cmd+V — terminal sends bracketed paste. Empty = image in clipboard."""
            pasted = event.data
            if pasted and pasted.strip():
                # Normal text paste
                event.current_buffer.insert_text(pasted)
            else:
                # Empty paste — check for image
                if clipboard_has_image():
                    img = read_clipboard_image()
                    if img and not any(e.base64_data == img.base64_data for e in self._pending_images):
                        self._pending_images.append(img)
                        import sys as _s
                        _s.stdout.write(f"\n\033[2m  image attached ({img.size_kb}KB)\033[0m\n")
                        _s.stdout.flush()

        self.prompt = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=GemCompleter(self.repo_root),
            complete_while_typing=False,
            complete_in_thread=True,
            key_bindings=kb,
            style=Style.from_dict({
                "bottom-toolbar": "noreverse #888888 bg:default",
                "": "bg:#2d2d2d",
                "prompt": "bg:#2d2d2d #ffffff",
            }),
        )
        self.approvals = ApprovalQueue(self.console, self.prompt)
        self.session_allows: set[str] = set()
        self._thinking_tick = 0
        self._pending_images: list[ImageData] = []
        self._pending_audio: list[AudioInputData] = []
        self.stats = SessionStats()
        self.tool_cache = ToolResultCache(self.repo_root)
        self.bg_indexer = None
        self.file_watcher = None
        self.project_watcher = None
        if not self.config.runtime.low_overhead_mode:
            self.bg_indexer = BackgroundIndexer(self.repo_root)
            self.bg_indexer.start()
            self.file_watcher = FileWatcher(self.repo_root)
            self.file_watcher.start()
            for f in self.session.pinned_files:
                self.file_watcher.track(f)
            self.project_watcher = ProjectWatcher(self.repo_root, poll_interval=3.0)
            self.project_watcher.on_change(self._on_project_files_changed)
            self.project_watcher.start()
        self._vim_mode = False
        self._active_plan: Plan | None = None
        self._output_style: str = ""
        self.task_store = TaskStore()
        self.planner_hints = PlannerHintState()
        self.out = OutputManager()  # centralized output
        self.out.set_event_callback(self._record_exec_event)
        self.logger = SessionLogger()
        self.history = HistoryDB()  # unified SQLite history
        self.embedding_search = EmbeddingSearch(str(self.repo_root))
        self.perms = PermissionManager()  # tool permissions
        self.hooks = HookRunner(str(self.repo_root), self.session.session_id, self.runtime_model)
        self.snapshot_store = SnapshotStore()
        self.turn_tracker = TurnDiffTracker(self.repo_root)
        # Apply autonomy mode
        autonomy_level = os.environ.get("GEM_AUTONOMY", "auto_edit")
        self._autonomy = AutonomyLevel(autonomy_level) if autonomy_level in ("suggest", "auto_edit", "full_auto") else AutonomyLevel.AUTO_EDIT
        apply_autonomy_to_permissions(self.perms, get_policy(self._autonomy))
        # Run session start hook
        self.hooks.on_session_start()
        self._spec_executor = SpeculativeExecutor(self.toolkit.execute_tool_calls)
        self._memory = self._load_memory()

    def _record_exec_event(self, event_type: str, payload: dict) -> None:
        detail = payload.get("stage") or payload.get("name") or payload.get("message") or payload.get("chunk") or event_type
        detail_text = str(detail).replace("\n", " ")[:200]
        self.store.append_event(
            self.session,
            f"exec:{event_type}",
            detail_text,
            **{k: str(v)[:240] for k, v in payload.items() if v is not None},
        )

    def _load_memory(self) -> dict:
        p = ensure_home_dirs() / "memory.json"
        if p.exists():
            try:
                import json
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    def _save_memory(self) -> None:
        import json
        p = ensure_home_dirs() / "memory.json"
        p.write_text(json.dumps(self._memory, indent=2))

    def _record_runtime_sample(self, *, first_token_s: float | None, total_s: float | None) -> None:
        if self.profile.key != "gemma4-26b-laptop":
            return
        mode = "fit" if self.config.runtime.provider == "llama_cpp" else "speed"
        bucket = self._memory.setdefault("runtime_telemetry", {}).setdefault("gemma4-26b-laptop", {}).setdefault(mode, {})
        samples = int(bucket.get("samples", 0)) + 1
        bucket["samples"] = samples
        alpha = 0.35
        if first_token_s is not None:
            prev = float(bucket.get("ema_first_token_s", first_token_s))
            bucket["ema_first_token_s"] = round(prev * (1 - alpha) + first_token_s * alpha, 3)
        if total_s is not None:
            prev_total = float(bucket.get("ema_total_s", total_s))
            bucket["ema_total_s"] = round(prev_total * (1 - alpha) + total_s * alpha, 3)
        bucket["provider"] = self.config.runtime.provider
        bucket["model"] = self.runtime_model
        bucket["updated_at"] = int(time.time())
        self._save_memory()

    def _on_project_files_changed(self, modified: list[str], created: list[str], deleted: list[str]) -> None:
        """Called by ProjectWatcher when files change externally."""
        # Invalidate tool cache for modified/deleted files
        for f in modified + deleted:
            self.tool_cache.invalidate(f)
        # Update embedding index incrementally if it exists
        if hasattr(self, 'embedding_search') and self.embedding_search.is_indexed():
            all_changed = modified + created + deleted
            if all_changed:
                try:
                    self.embedding_search.update_files(all_changed)
                except Exception:
                    pass

    def run(self) -> None:
        self.console.print(self._welcome_view())
        # Mode picker right after banner
        if getattr(self, "_show_mode_picker", False):
            self._pick_mode()
        else:
            self.console.print()
        # Show last messages if resuming a session
        if len(self.session.messages) > 0:
            recent = self.session.messages[-4:]
            self.console.print("[dim]  --- resumed session ---[/]")
            for msg in recent:
                role = msg.get("role", "")
                content = str(msg.get("content", ""))[:120]
                if role == "user":
                    self.console.print(f"[dim]  > {content}[/]")
                elif role == "assistant":
                    self.console.print(f"[dim]  {content}[/]")
            self.console.print("[dim]  --- end ---[/]\n")
        # Warm up model with pulsating dots
        import sys as _sys, threading, termios, time as _time
        # Disable echo during loading
        try:
            fd = _sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        except Exception:
            fd = None
            old_settings = None
        # Pulsating dots animation
        _loading = True
        def _animate():
            dots = 0
            while _loading:
                dots = (dots % 4) + 1
                _sys.stderr.write(f"\r\033[32m  loading model{'.' * dots}{' ' * (4 - dots)}\033[0m")
                _sys.stderr.flush()
                _time.sleep(0.3)
        anim = threading.Thread(target=_animate, daemon=True)
        anim.start()
        try:
            # Auto-start server if not running
            import httpx as _httpx
            server_ok = False
            try:
                r = _httpx.get(f"{self.config.runtime.base_url.rstrip('/')}/health", timeout=3)
                server_ok = r.status_code == 200
            except Exception:
                pass
            if not server_ok and self.config.runtime.provider == "llama_cpp":
                from .runtime_launch import launch_runtime
                _sys.stderr.write(f"\r\033[33m  starting server...{' ' * 20}\033[0m")
                _sys.stderr.flush()
                launch_runtime(self.config.runtime, self.repo_root)
                # Wait for server to be ready
                for _ in range(60):
                    _time.sleep(1)
                    try:
                        r = _httpx.get(f"{self.config.runtime.base_url.rstrip('/')}/health", timeout=2)
                        if r.status_code == 200:
                            server_ok = True
                            break
                    except Exception:
                        pass
                if not server_ok:
                    _sys.stderr.write(f"\r\033[31m  server failed to start{' ' * 20}\033[0m\n")

            _loading = False
            anim.join(timeout=1)
            _sys.stderr.write(f"\r\033[32m  model ready ✓       \033[0m\n\n")
        except Exception as exc:
            _loading = False
            anim.join(timeout=1)
            _sys.stderr.write(f"\r\033[33m  {exc}\033[0m\n\n")
        _sys.stderr.flush()
        if fd is not None and old_settings is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                termios.tcflush(fd, termios.TCIFLUSH)
            except Exception:
                pass
        while True:
            try:
                # Check for external file changes
                changes = self.file_watcher.get_changes() if self.file_watcher is not None else []
                if changes:
                    self.tool_cache.invalidate_all()
                    self.console.print(f"  [dim yellow]files changed externally: {', '.join(changes[:5])}[/]")
                import sys
                sys.stdout.write("\033[5 q")  # blinking bar cursor
                sys.stdout.flush()
                raw = self.prompt.prompt(
                    self._prompt_label(),
                    bottom_toolbar=self._bottom_toolbar,
                ).strip()
                sys.stdout.write("\033[0 q")  # restore default cursor
                sys.stdout.flush()
            except (EOFError, KeyboardInterrupt):
                self.out.done()
                self.console.print("\nExiting.")
                return
            if not raw:
                # Empty input after paste = image-only message
                if self._pending_images:
                    raw = "What's in this image?"
                else:
                    continue
            if raw.startswith("/"):
                should_continue = self._handle_command(raw)
                if not should_continue:
                    return
                continue
            # Auto-detect image file paths in the message
            self._detect_inline_images(raw)
            self._chat_turn(raw)

    def _prompt_label(self) -> str:
        parts = []
        if self._pending_images:
            count = len(self._pending_images)
            total_kb = sum(img.size_kb for img in self._pending_images)
            parts.append(f"{count} img {total_kb}KB")
        if self._pending_audio:
            count = len(self._pending_audio)
            total_sec = sum(a.duration_seconds for a in self._pending_audio)
            parts.append(f"{count} audio {total_sec:.0f}s")
        if parts:
            return f"> ({', '.join(parts)}) "
        return "> "

    def _bottom_toolbar(self) -> str:
        """Status bar: model · context left · path."""
        chars_used = sum(len(str(m.get("content", ""))) for m in self.session.messages)
        max_chars = self._effective_context_chars()
        pct_left = max(0, 100 - int(chars_used / max(1, max_chars) * 100))
        mode = self.config.runtime.laptop_26b_runtime_mode
        mode_label = "fast" if not mode.endswith("-think") else "reasoning"
        model_short = self.runtime_model.split("/")[-1] if "/" in self.runtime_model else self.runtime_model
        path = f"~/{self.repo_root.name}" if self.repo_root != Path.home() else "~"
        return f"\n {model_short} {mode_label} · {pct_left}% left · {path}"

    def _check_clipboard_image(self) -> None:
        """Silently check if clipboard has a new image."""
        if not self.profile.supports_vision:
            return
        if clipboard_has_image():
            img = read_clipboard_image()
            if img and not any(existing.base64_data == img.base64_data for existing in self._pending_images):
                self._pending_images.append(img)
                self.console.print(f"  [dim cyan]image detected in clipboard ({img.size_kb}KB) — will be sent with your next message[/]")

    def _detect_inline_images(self, text: str) -> None:
        """Detect image and audio file paths mentioned in the user's text."""
        if self.profile.supports_vision:
            paths = detect_image_paths_in_text(text)
            for path in paths:
                img = load_image_from_path(path)
                if img:
                    self._pending_images.append(img)
                    self.console.print(f"  [dim cyan]attached image: {path} ({img.size_kb}KB)[/]")
        # Auto-detect audio files
        if self.profile.supports_audio:
            audio_paths = detect_audio_paths_in_text(text)
            for path in audio_paths:
                aud = load_audio(path)
                if aud:
                    self._pending_audio.append(aud)
                    self.console.print(f"  [dim cyan]attached audio: {path} ({aud.duration_str})[/]")

    def _banner(self) -> str:
        return format_banner(
            repo_root=str(self.repo_root),
            session_id=self.session.session_id,
            profile_name=self.profile.display_name,
            model_name=self.runtime_model,
        )

    def _welcome_view(self):
        from rich.align import Align
        banner_text = self._banner()
        return Panel(
            Align.center(banner_text),
            title="[bold]◆ LOCALcode[/]",
            border_style="green",
            style="green",
            expand=True,
        )

    def _pick_mode(self) -> None:
        """Mode picker shown after banner."""
        import subprocess, sys, tty, termios
        # Auto-unlock GPU if needed
        try:
            r = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                               capture_output=True, text=True, timeout=2)
            gpu_ok = int(r.stdout.strip()) >= 14000
        except Exception:
            gpu_ok = False
        if not gpu_ok:
            print("  GPU needs a one-time unlock (resets on reboot).")
            if input("  Allow? [Y/n]: ").strip().lower() in ("", "y", "yes"):
                subprocess.run(["sudo", "sysctl", "iogpu.wired_limit_mb=14336"])
                gpu_ok = True
        if gpu_ok:
            print()
            print("  1. \033[1mFast\033[0m         27 tok/s  32K context")
            print("  2. \033[1mReasoning\033[0m    26 tok/s  32K context")
            ch = ""
            # Try single-keypress read, fall back to input()
            try:
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    sys.stdout.write("  > ")
                    sys.stdout.flush()
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                try:
                    ch = input("  > ").strip()
                except EOFError:
                    ch = "1"
            sys.stdout.write(f"\r                    \n")
            if ch == "2":
                self.config.runtime.laptop_26b_runtime_mode = "turbo-think"
                sys.stdout.write(f"\033[32m  reasoning ✓\033[0m\n")
            else:
                self.config.runtime.laptop_26b_runtime_mode = "turbo"
                sys.stdout.write(f"\033[32m  fast ✓\033[0m\n")
            sys.stdout.flush()
        else:
            self.config.runtime.laptop_26b_runtime_mode = "speed"

    def _confirm(self, prompt: str, default: bool = False) -> bool:
        suffix = " [Y/n] " if default else " [y/N] "
        raw = self.prompt.prompt(prompt + suffix).strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}

    def render_agent_state(self, phase: str, detail: str) -> None:
        style_map = {
            "plan": "cyan",
            "step": "bright_cyan",
            "verify": "yellow",
            "repair": "red",
            "escalation": "magenta",
            "done": "green",
        }
        style = style_map.get(phase, "dim")
        self.console.print(Panel.fit(
            f"  [{style}]{detail}[/]",
            title=f"[bold {style}]agent > {phase}[/]",
            border_style=style,
        ))

    def _handle_command(self, raw: str) -> bool:
        name, _, arg = raw.partition(" ")
        arg = arg.strip()
        if name == "/help":
            self.console.print(
                Panel.fit(
                    "[bold]Navigation[/]\n"
                    "  /help /status /model /files /find /index /read /add /drop /context /diff\n\n"
                    "[bold]Editing[/]\n"
                    "  /apply /undo [all] /changes /shell <cmd> /bg <cmd> /verify [cmd]\n\n"
                    "[bold]Agent[/]\n"
                    "  /agent <task> /agentbg <task>\n\n"
                    "[bold]Tools[/]\n"
                    "  /tools /skills /mcp /permissions /search <query> /browser /voice\n\n"
                    "[bold]Session[/]\n"
                    "  /thinking [hidden|summary|full] /timeline /jobs /log <id> /clear /quit",
                    title="[bold bright_cyan]gem commands[/]",
                    border_style="bright_cyan",
                )
            )
            return True
        if name == "/status":
            table = Table(show_header=False)
            table.add_row("repo", str(self.repo_root))
            table.add_row("session", self.session.session_id)
            table.add_row("profile", self.profile.display_name)
            table.add_row("variant", self.profile.feature_variant)
            table.add_row("family", self.profile.family)
            table.add_row("model", self.runtime_model)
            table.add_row("mode", self.config.runtime.mode)
            table.add_row("planner_enabled", str(self.config.runtime.planner_enabled))
            table.add_row("planner_model", self.config.runtime.planner_model)
            table.add_row("adaptive_execution", str(self.config.runtime.adaptive_execution))
            table.add_row("thinking", self.config.ui.thinking_mode)
            table.add_row("agent_steps", str(self.profile.agent_steps))
            table.add_row("retrieval_budget", str(self.profile.retrieval_budget))
            table.add_row("verification_bias", self.profile.verification_bias)
            provider, search_status = self.toolkit.search_status()
            table.add_row("search", f"{provider} ({search_status})")
            table.add_row("browser", self.config.browser.mcp_server_name if self.config.browser.enabled else "disabled")
            table.add_row("voice", f"{self.config.voice.stt_provider} + {self.config.voice.tts_provider}")
            table.add_row("git", git_status(self.repo_root))
            table.add_row("pinned", ", ".join(self.session.pinned_files) or "(none)")
            self.console.print(table)
            diagnostics = self.toolkit.diagnostics()
            if diagnostics:
                self.console.print(Panel("\n".join(diagnostics), title="Diagnostics"))
            self.console.print(
                Panel.fit(
                    "\n".join(
                        [
                            "Shell, patch apply, background jobs, and model-driven tool calls require explicit approval.",
                            "Use /tools to inspect available tools and /mcp to inspect connected MCP servers.",
                            "Use /permissions to persist repo-scoped allow or deny rules.",
                        ]
                    ),
                    title="Safety",
                )
            )
            return True
        if name == "/model":
            if not arg:
                table = Table("profile", "default_model", "variant", "summary")
                for profile in GEMMA_PROFILES.values():
                    table.add_row(profile.key, profile.default_model, profile.feature_variant, profile.summary)
                self.console.print(table)
                self.console.print(f"Active profile: {self.profile.key}")
                self.console.print(f"Active model: {self.runtime_model}")
                try:
                    installed = self.engine.list_models()
                    self.console.print("Installed Ollama models: " + (", ".join(installed) or "(none)"))
                except Exception:
                    self.console.print("Installed Ollama models: unavailable")
                return True
            selected_profile = GEMMA_PROFILES.get(arg.lower()) or infer_profile_from_model(arg)
            if selected_profile and arg.lower() in GEMMA_PROFILES:
                self.profile = selected_profile
                self.runtime_model = selected_profile.default_model
            else:
                self.profile = selected_profile or self.profile
                self.runtime_model = arg
            self.session.profile = self.profile.key
            self.session.model = self.runtime_model
            self.config.runtime.profile = self.profile.key
            self.config.runtime.model = self.runtime_model
            self.engine.close()
            self.engine = GemRuntimeGateway(self.config.runtime)
            save_config(self.config)
            self.store.append_event(self.session, "model_switch", f"{self.profile.key} {self.runtime_model}")
            self.store.save(self.session)
            self.console.print(f"  Switched to [green]{self.runtime_model}[/] ({self.profile.display_name})")
            # Warm up new model in background
            import threading
            def _warm():
                try:
                    self.engine.chat_once([{"role": "user", "content": "hi"}])
                except Exception:
                    pass
            threading.Thread(target=_warm, daemon=True).start()
            self.console.print(f"  [dim]Warming up model...[/]")
            return True
        if name == "/skills":
            self.console.print("\n".join(list_skills(self.repo_root)) or "No skills found.")
            return True
        if name == "/tools":
            self.console.print("\n".join(self.toolkit.list_tool_names()) or "No tools loaded.")
            return True
        if name == "/mcp":
            self.toolkit.ensure_mcp_tools()
            configs = load_mcp_configs()
            if not configs:
                self.console.print("No MCP servers configured.")
                return True
            table = Table("name", "command", "args")
            for cfg in configs:
                table.add_row(cfg.name, cfg.command, " ".join(cfg.args))
            self.console.print(table)
            return True
        if name == "/permissions":
            table = Table("tool", "rule")
            for tool_name, rule in self.permissions.status_rows():
                table.add_row(tool_name, rule)
            self.console.print(table)
            return True
        if name == "/timeline":
            table = Table("time", "type", "detail")
            for event in self.session.events[-30:]:
                table.add_row(event.get("time", ""), event.get("type", ""), event.get("detail", ""))
            self.console.print(table)
            return True
        if name == "/thinking":
            if not arg:
                self.console.print(f"Thinking mode: {self.config.ui.thinking_mode}")
                return True
            if arg not in {"hidden", "summary", "full"}:
                self.console.print("Thinking mode must be one of: hidden, summary, full")
                return True
            self.config.ui.thinking_mode = arg
            save_config(self.config)
            self.store.append_event(self.session, "thinking_mode", arg)
            self.console.print(f"Thinking mode set to {arg}")
            return True
        if name == "/search":
            if not arg:
                self.console.print("Usage: /search <query>")
                return True
            # Try semantic code search first
            if self.embedding_search.is_indexed():
                results = self.embedding_search.search(arg, top_k=5)
                if results:
                    table = Table("file", "lines", "score", "preview")
                    for r in results:
                        table.add_row(r.file, f"{r.start_line}-{r.end_line}", f"{r.score:.2f}", r.preview[:60])
                    self.console.print(table)
                    return True
            # Fall back to web search
            self.console.print(Panel(self.toolkit._web_search(arg), title=f"Search: {arg}"))
            return True
        if name == "/history":
            if arg == "stats":
                stats = self.history.get_stats(str(self.repo_root))
                table = Table("metric", "value")
                table.add_row("prompts", str(stats["prompts"]))
                table.add_row("responses", str(stats["responses"]))
                table.add_row("tool calls", str(stats["tool_calls"]))
                table.add_row("sessions", str(stats["sessions"]))
                table.add_row("tokens (in)", f"{stats['total_tokens_in']:,}")
                table.add_row("tokens (out)", f"{stats['total_tokens_out']:,}")
                table.add_row("total time", f"{stats['total_duration_s']}s")
                if stats["top_tools"]:
                    table.add_row("top tools", ", ".join(f"{t}({c})" for t, c in stats["top_tools"][:5]))
                if stats["top_files"]:
                    table.add_row("top files", ", ".join(f"{f}({c})" for f, c in stats["top_files"][:5]))
                self.console.print(table)
            elif arg:
                # Search history
                results = self.history.search_history(arg, str(self.repo_root), limit=10)
                if results:
                    table = Table("role", "content", "time")
                    for r in results:
                        from datetime import datetime
                        ts = datetime.fromtimestamp(r["timestamp"]).strftime("%m/%d %H:%M")
                        table.add_row(r["role"], r["content"][:80], ts)
                    self.console.print(table)
                else:
                    self.console.print(f"No history matching: {arg}")
            else:
                # Show recent prompts
                recent = self.history.get_recent_prompts(str(self.repo_root), limit=15)
                if recent:
                    table = Table("time", "prompt")
                    for r in recent:
                        from datetime import datetime
                        ts = datetime.fromtimestamp(r["timestamp"]).strftime("%m/%d %H:%M")
                        table.add_row(ts, r["content"][:80])
                    self.console.print(table)
                else:
                    self.console.print("No history yet.")
            return True
        if name == "/autonomy":
            if not arg:
                self.console.print(format_autonomy_status(self._autonomy))
                return True
            if arg in ("suggest", "auto_edit", "full_auto"):
                self._autonomy = AutonomyLevel(arg)
                apply_autonomy_to_permissions(self.perms, get_policy(self._autonomy))
                self.console.print(f"Autonomy set to: {arg}")
            else:
                self.console.print("Usage: /autonomy [suggest|auto_edit|full_auto]")
            return True
        if name == "/snapshot":
            if not arg or arg == "list":
                snaps = self.snapshot_store.list_for_repo(str(self.repo_root))
                if snaps:
                    table = Table("id", "reason", "messages", "time")
                    for s in snaps[:10]:
                        from datetime import datetime
                        ts = datetime.fromtimestamp(s.created_at).strftime("%m/%d %H:%M")
                        table.add_row(s.id[:8], s.reason, str(len(s.messages)), ts)
                    self.console.print(table)
                else:
                    self.console.print("No snapshots saved.")
                return True
            if arg.startswith("restore "):
                snap_id = arg.split(" ", 1)[1].strip()
                snap = self.snapshot_store.load(snap_id)
                if snap:
                    self.session.messages = list(snap.messages)
                    self.session.pinned_files = list(snap.pinned_files)
                    restored = restore_snapshot(snap)
                    self.store.save(self.session)
                    self.console.print(f"Restored snapshot {snap_id[:8]}: {len(snap.messages)} messages, {len(restored)} files")
                else:
                    self.console.print(f"Snapshot not found: {snap_id}")
                return True
            if arg == "save":
                snap = create_snapshot(
                    self.session.session_id, str(self.repo_root),
                    self.session.messages, self.session.pinned_files,
                    model=self.runtime_model, reason="manual",
                )
                self.console.print(f"Snapshot saved: {snap.id[:8]}")
                return True
            return True
        if name == "/browser":
            if not arg or arg == "status":
                self.console.print(Panel("\n".join(browser_status(self.config)), title="Browser"))
                return True
            if arg == "setup":
                path = ensure_browser_mcp(self.config)
                self.toolkit.close()
                self.toolkit = GemToolkit(self.repo_root, self.config)
                self.console.print(f"Browser MCP preset saved to {path}")
                return True
            if arg.startswith("open "):
                self.toolkit.ensure_mcp_tools()
                url = arg[5:].strip()
                if not url:
                    self.console.print("Usage: /browser open <url>")
                    return True
                tool_name = find_browser_tool(self.toolkit.tools, "open")
                if not tool_name:
                    self.console.print("No browser navigation tool is loaded. Run `/browser setup` and restart Gem.")
                    return True
                if not self._approve_action("browser", f"Open browser page?\n{url}"):
                    self.console.print("Cancelled.")
                    return True
                result = self.toolkit.tools[tool_name].handler({"url": url})
                self.store.append_event(self.session, "browser_open", url)
                self.console.print(Panel(result[-4000:], title="Browser"))
                return True
            if arg == "snapshot":
                self.toolkit.ensure_mcp_tools()
                tool_name = find_browser_tool(self.toolkit.tools, "snapshot")
                if not tool_name:
                    self.console.print("No browser snapshot tool is loaded. Run `/browser setup` and restart Gem.")
                    return True
                if not self._approve_action("browser", "Capture a browser accessibility snapshot?"):
                    self.console.print("Cancelled.")
                    return True
                result = self.toolkit.tools[tool_name].handler({})
                self.store.append_event(self.session, "browser_snapshot", "snapshot")
                self.console.print(Panel(result[-4000:], title="Browser Snapshot"))
                return True
            self.console.print("Usage: /browser [setup|status|open <url>|snapshot]")
            return True
        if name == "/voice":
            if not arg or arg == "status":
                self.console.print(Panel("\n".join(voice_status(self.config)), title="Voice"))
                return True
            if arg.startswith("say "):
                text = arg[4:].strip()
                if not text:
                    self.console.print("Usage: /voice say <text>")
                    return True
                result = speak_text(self.config, text)
                self.store.append_event(self.session, "voice_say", text[:120])
                self.console.print(Panel(result, title="Voice"))
                return True
            if arg.startswith("transcribe "):
                file_arg = arg[11:].strip()
                if not file_arg:
                    self.console.print("Usage: /voice transcribe <file>")
                    return True
                result = transcribe_audio(self.config, file_arg)
                self.store.append_event(self.session, "voice_transcribe", file_arg)
                self.console.print(Panel(result[-4000:], title="Voice"))
                return True
            self.console.print("Usage: /voice [status|say <text>|transcribe <file>]")
            return True
        if name == "/verify":
            plan = build_verification_plan(self.repo_root, bias=self.profile.verification_bias)
            if not arg:
                self.console.print(
                    Panel(
                        "\n".join(f"{step.label}: {step.command}" for step in plan) or "No verification plan detected.",
                        title="Verification Plan",
                    )
                )
            output, code = run_verification(self.repo_root, arg or None, bias=self.profile.verification_bias)
            self.store.append_event(self.session, "verify", f"exit={code} {arg or guess_verify_command(self.repo_root) or 'none'}")
            self.console.print(Panel(output[-6000:], title=f"Verification exit={code}"))
            return True
        if name == "/agentbg":
            job_id = launch_background_agent(arg, self.repo_root, self.profile.key, self.runtime_model)
            self.console.print(f"Started background agent job {job_id}")
            return True
        if name == "/ignite":
            self.console.print("Use `gem setup --install` before first launch if the local runtime is not ready.")
            return True
        if name == "/files":
            files = list_repo_files(self.repo_root, pattern=arg or None)
            self.console.print("\n".join(files) if files else "No files matched.")
            return True
        if name == "/index":
            count, path = build_index(self.repo_root)
            self.console.print(f"Built text index for {count} files at {path}")
            # Also build embedding index
            if self.embedding_search.setup():
                embed_count = self.embedding_search.build_index()
                self.console.print(f"Built embedding index: {embed_count} chunks")
            else:
                self.console.print("[dim]Embedding search unavailable (install sentence-transformers for semantic search)[/dim]")
            return True
        if name == "/find":
            results = search_index(self.repo_root, arg)
            if not results:
                count, path = build_index(self.repo_root)
                self.console.print(f"Built code index for {count} files at {path}")
                results = search_index(self.repo_root, arg)
            table = Table("path", "chunk", "preview")
            for item in results:
                table.add_row(item["path"], item["chunk_id"], item["preview"])
            self.console.print(table)
            return True
        if name == "/read":
            self.console.print(Panel(read_file(self.repo_root, arg), title=arg))
            return True
        if name == "/add":
            if arg and arg not in self.session.pinned_files:
                self.session.pinned_files.append(arg)
                self.store.save(self.session)
            self.console.print(f"Pinned: {arg}")
            return True
        if name == "/drop":
            self.session.pinned_files = [item for item in self.session.pinned_files if item != arg]
            self.store.save(self.session)
            self.console.print(f"Removed: {arg}")
            return True
        if name == "/context":
            context = build_context_block(
                self.repo_root,
                self.session.pinned_files,
                min(self.config.runtime.max_context_chars, self.profile.recommended_context_chars),
            )
            self.console.print(Panel(context, title="Context"))
            return True
        if name == "/shell":
            if not arg:
                self.console.print("Usage: /shell <command>")
                return True
            if not self._approve_action("shell", f"Run shell command in {self.repo_root}?\n{arg}"):
                self.console.print("Cancelled.")
                return True
            self.console.print(Panel.fit(f"$ {arg}", title="Shell"))
            result = run_shell(arg, str(self.repo_root), on_output=self.console.print)
            self.console.print(f"[exit {result.returncode}]")
            self.session.messages.append(
                {
                    "role": "user",
                    "content": f"Shell command executed: {arg}\n\nOutput:\n{result.output}",
                }
            )
            self.store.append_event(self.session, "shell", arg)
            self.store.save(self.session)
            return True
        if name == "/bg":
            if not arg:
                self.console.print("Usage: /bg <command>")
                return True
            if not self._approve_action("background_job", f"Start background job in {self.repo_root}?\n{arg}"):
                self.console.print("Cancelled.")
                return True
            job = launch_background_job(arg, self.repo_root)
            self.store.append_event(self.session, "background_job", f"{job.job_id} {arg}")
            self.console.print(f"Started background job {job.job_id}")
            return True
        if name == "/jobs":
            table = Table("job_id", "status", "command", "created_at")
            for row in list_jobs():
                table.add_row(row["job_id"], row["status"], row["command"], row["created_at"])
            self.console.print(table)
            return True
        if name == "/log":
            self.console.print(Panel(read_job_log(arg), title=f"Job {arg}"))
            return True
        if name == "/diff":
            self.console.print(Panel(git_diff(self.repo_root), title="Git Diff"))
            return True
        if name == "/apply":
            diff_block = extract_last_diff_block(self.session.last_assistant_text)
            if not diff_block:
                self.console.print("No assistant diff block found.")
                return True
            reviewed_diff = self._review_patch(diff_block)
            if not reviewed_diff:
                self.console.print("Cancelled.")
                return True
            if not self._approve_action("patch_apply", "Apply the reviewed patch with git apply?"):
                self.console.print("Cancelled.")
                return True
            ok, output = apply_diff(self.repo_root, reviewed_diff)
            title = "Patch Applied" if ok else "Patch Failed"
            self.store.append_event(self.session, "patch_apply", title)
            self.console.print(Panel(output, title=title))
            return True
        if name == "/image":
            if not self.profile.supports_vision:
                self.console.print("Current model profile does not support vision.")
                return True
            if not arg:
                if self._pending_images:
                    self.console.print(f"  {len(self._pending_images)} image(s) queued ({sum(i.size_kb for i in self._pending_images)}KB total)")
                    for i, img in enumerate(self._pending_images):
                        self.console.print(f"    [{i}] {img.source} ({img.size_kb}KB)")
                else:
                    self.console.print("No images queued. Use /image <path>, /paste, or /screenshot.")
                return True
            if arg == "clear":
                self._pending_images.clear()
                self.console.print("Cleared queued images.")
                return True
            # Load image from path
            img = load_image_from_path(arg)
            if img:
                self._pending_images.append(img)
                self.console.print(f"  [cyan]Queued image: {arg} ({img.size_kb}KB)[/]")
            else:
                self.console.print(f"Could not load image: {arg}")
            return True
        if name == "/paste":
            if not self.profile.supports_vision:
                self.console.print("Current model profile does not support vision.")
                return True
            img = read_clipboard_image()
            if img:
                self._pending_images.append(img)
                self.console.print(f"  [cyan]Pasted image from clipboard ({img.size_kb}KB)[/]")
            else:
                self.console.print("No image found in clipboard.")
            return True
        if name == "/screenshot":
            if not self.profile.supports_vision:
                self.console.print("Current model profile does not support vision.")
                return True
            mode = "selection" if arg == "select" else "full"
            self.console.print(f"  [dim]Capturing {mode} screenshot...[/]")
            img = take_screenshot(mode)
            if img:
                self._pending_images.append(img)
                self.console.print(f"  [cyan]Screenshot captured ({img.size_kb}KB)[/]")
            else:
                self.console.print("Screenshot capture failed.")
            return True
        if name == "/audio":
            if not arg:
                if self._pending_audio:
                    for i, aud in enumerate(self._pending_audio):
                        self.console.print(f"  [{i}] {aud.source} ({aud.duration_str}, ~{aud.estimated_tokens} tokens)")
                else:
                    self.console.print("No audio queued. Use /audio <path>")
                return True
            if arg == "clear":
                self._pending_audio.clear()
                self.console.print("Cleared queued audio.")
                return True
            aud = load_audio(arg)
            if aud:
                self._pending_audio.append(aud)
                self.console.print(f"  [cyan]Queued audio: {arg} ({aud.duration_str}, ~{aud.estimated_tokens} tokens)[/]")
            else:
                self.console.print(f"Could not load audio: {arg}")
            return True
        if name == "/undo":
            if arg == "all":
                messages = self.toolkit.changes.undo_all()
                for msg in messages:
                    self.console.print(f"  {msg}")
                self.console.print(f"Undid {len(messages)} changes.")
            else:
                ok, msg = self.toolkit.changes.undo_last()
                self.console.print(msg)
            return True
        if name == "/changes":
            rows = self.toolkit.changes.status()
            if not rows:
                self.console.print("No tracked changes.")
                return True
            table = Table("time", "action", "tool", "path")
            for row in rows:
                table.add_row(row["time"], row["action"], row["tool"], row["path"])
            self.console.print(table)
            return True
        if name == "/clear":
            self.session.messages.clear()
            self.session.last_assistant_text = ""
            self.store.save(self.session)
            self.console.print("Conversation cleared.")
            return True
        if name == "/tasks":
            if not arg:
                self.console.print(self.task_store.list_all())
            elif arg.startswith("add "):
                t = self.task_store.create(arg[4:].strip())
                self.console.print(f"  Created {t.id}: {t.title}")
            elif arg.startswith("done "):
                self.console.print(self.task_store.update(arg[5:].strip(), status="done"))
            elif arg == "clear":
                n = self.task_store.clear_done()
                self.console.print(f"  Cleared {n} completed tasks")
            else:
                self.console.print("  /tasks [add <title> | done <id> | clear]")
            return True
        if name == "/memory":
            if not arg:
                self.console.print("  /memory set <key> <value> — save a preference")
                self.console.print("  /memory get <key> — recall a preference")
                self.console.print("  /memory list — show all preferences")
            elif arg.startswith("set "):
                parts = arg[4:].strip().split(" ", 1)
                if len(parts) == 2:
                    self._memory[parts[0]] = parts[1]
                    self._save_memory()
                    self.console.print(f"  Saved: {parts[0]} = {parts[1]}")
            elif arg.startswith("get "):
                key = arg[4:].strip()
                val = self._memory.get(key, "(not set)")
                self.console.print(f"  {key} = {val}")
            elif arg == "list":
                for k, v in self._memory.items():
                    self.console.print(f"  {k} = {v}")
            return True
        if name == "/replay":
            # Session playback
            for msg in self.session.messages[-20:]:
                role = msg.get("role", "")
                content = str(msg.get("content", ""))[:100]
                if role == "user":
                    self.console.print(f"  [green]> {content}[/]")
                elif role == "assistant":
                    self.console.print(f"  [dim]{content}[/]")
            return True
        if name == "/config":
            if not arg:
                self.console.print(f"  provider: {self.config.runtime.provider}")
                self.console.print(f"  model: {self.config.runtime.model}")
                self.console.print(f"  temperature: {self.config.runtime.temperature}")
                self.console.print(f"  mode: {self.config.runtime.mode}")
            elif arg.startswith("set "):
                parts = arg[4:].strip().split(" ", 1)
                if len(parts) == 2:
                    key, val = parts
                    from .settings import set_setting
                    result = set_setting(self.config, key, val)
                    self.console.print(f"  {result}")
            return True
        if name == "/export":
            # Export conversation as markdown
            import time as _t
            filename = arg or f"jem-export-{_t.strftime('%Y%m%d-%H%M%S')}.md"
            lines = [f"# Jem Session Export\n", f"Session: {self.session.session_id}\n\n"]
            for msg in self.session.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"## > {content}\n\n")
                elif role == "assistant":
                    lines.append(f"{content}\n\n---\n\n")
                elif role == "tool":
                    lines.append(f"*Tool result:* {content[:200]}\n\n")
            path = self.repo_root / filename
            path.write_text("".join(lines))
            self.console.print(f"  Exported to {path}")
            return True
        if name == "/style":
            styles = {
                "concise": "Be extremely concise. One-line answers when possible. No preamble.",
                "detailed": "Give thorough, detailed explanations with examples.",
                "code-only": "Respond with code only. No explanations unless asked.",
            }
            if not arg or arg not in styles:
                self.console.print(f"  Styles: {', '.join(styles.keys())}")
                return True
            self._output_style = styles[arg]
            self.console.print(f"  Style set to: {arg}")
            return True
        if name == "/branch":
            import copy, json as _json
            branch_dir = self.store.path.parent / "branches"
            branch_dir.mkdir(exist_ok=True)
            if not arg or arg == "save":
                branch_id = f"branch-{len(list(branch_dir.glob('*.json')))}"
                data = {"messages": copy.deepcopy(self.session.messages), "id": branch_id}
                (branch_dir / f"{branch_id}.json").write_text(_json.dumps(data))
                self.console.print(f"  Saved branch: {branch_id}")
            elif arg == "list":
                for f in sorted(branch_dir.glob("*.json")):
                    self.console.print(f"  {f.stem}")
            elif arg.startswith("checkout "):
                bid = arg.split(" ", 1)[1]
                bp = branch_dir / f"{bid}.json"
                if bp.exists():
                    data = _json.loads(bp.read_text())
                    self.session.messages = data["messages"]
                    self.store.save(self.session)
                    self.console.print(f"  Restored branch: {bid}")
                else:
                    self.console.print(f"  Branch not found: {bid}")
            else:
                self.console.print("  /branch [save|list|checkout <id>]")
            return True
        if name == "/vim":
            self._vim_mode = not self._vim_mode
            self.prompt.editing_mode = get_editing_mode(self._vim_mode)
            self.console.print(f"VIM mode: {'on' if self._vim_mode else 'off'}")
            return True
        if name == "/stats":
            self.console.print(f"  {self.stats.summary()}")
            self.console.print(f"  Cache entries: {self.tool_cache.size}")
            return True
        if name == "/orch":
            if not arg:
                self.console.print("Usage: /orch <task>  — run multi-agent orchestrated execution")
                return True
            from .orchestrator import Orchestrator
            self.out.start_thinking()
            plan = Orchestrator(self).run(arg)
            self.out.done()
            self.console.print(f"\n  [bold]{plan.summary}[/bold]")
            if plan.review_notes:
                self.console.print(f"  [dim]{plan.review_notes}[/dim]")
            for step in plan.steps:
                icon = {"done": "✓", "error": "✗", "skipped": "○"}.get(step.status, "·")
                self.console.print(f"    {icon} [{step.id}] {step.description[:70]}")
            return True
        if name == "/plan":
            if not arg:
                if self._active_plan:
                    self.console.print(self._active_plan.summary())
                else:
                    self.console.print("No active plan. Use /plan <task> to create one.")
                return True
            if arg == "show" and self._active_plan:
                self.console.print(self._active_plan.summary())
                return True
            if arg == "cancel":
                self._active_plan = None
                self.console.print("Plan cancelled.")
                return True
            if arg in ("go", "next") and self._active_plan:
                plan = self._active_plan
                while not plan.is_done:
                    step = plan.next_step
                    if step is None:
                        break
                    step.status = "running"
                    self.console.print(f"\n  [green]◆[/] Step {plan.current_step + 1}: {step.description}")
                    result = self.ask(f"Execute this step: {step.description}", stream=True)
                    step.result = result
                    step.status = "done"
                    plan.current_step += 1
                    if arg == "next":
                        break
                self.console.print(f"\n{plan.summary()}")
                if plan.is_done:
                    self._active_plan = None
                return True
            # Create a new plan
            self.console.print(f"  [dim]Planning: {arg}[/]")
            response = self.ask(
                f"Create a step-by-step plan for this task. List numbered steps. Do NOT execute yet, just plan.\n\nTask: {arg}",
                stream=True,
            )
            steps = parse_plan_from_response(response)
            if steps:
                self._active_plan = Plan(task=arg, steps=steps)
                self.console.print(f"\n{self._active_plan.summary()}")
                self.console.print(f"\n  [dim]Use /plan go to execute, /plan next for one step, /plan cancel to abort.[/]")
            else:
                self.console.print("Could not parse steps from response.")
            return True
        if name == "/commit":
            # Auto-commit: stage all, generate message from diff, commit
            self._chat_turn("Look at the current git diff and create a concise commit. Use git_status, then git_diff, then git_commit with an appropriate message. Do it now.")
            return True
        if name == "/review":
            # Review current changes
            self._chat_turn("Review the current git diff for bugs, security issues, and code quality. Use git_diff to see the changes, then give a thorough code review.")
            return True
        if name == "/brief":
            # Generate project brief
            self._chat_turn("Give me a brief summary of this project. Use list_files and read_file on key files like README, package.json, pyproject.toml to understand the project structure and purpose. Be concise.")
            return True
        if name == "/security":
            # Run security scan
            result = self.toolkit._security_scan(arg or "")
            self.console.print(result)
            return True
        if name == "/test":
            # Run tests
            result = self.toolkit._run_tests(command=arg or "")
            self.console.print(result)
            return True
        if name == "/quit":
            return False
        self.console.print(f"Unknown command: {name}")
        return True

    def _chat_turn(self, user_text: str) -> None:
        # Grab any pending images and clear the queue
        images = self._pending_images.copy()
        self._pending_images.clear()
        audio = self._pending_audio.copy()
        self._pending_audio.clear()
        self.ask(user_text, stream=True, images=images, audio=audio)

    def ask(self, user_text: str, stream: bool = True, images: list[ImageData] | None = None, audio: list[AudioInputData] | None = None) -> str:
        # Reset per-turn state
        self.perms.new_turn()

        # ── Dynamic task name (like Codex) ──
        task_name = self._generate_task_name(user_text)
        if stream and task_name:
            self.out.set_stage(task_name)

        # Start centralized output — thinking indicator appears immediately
        if stream:
            self.out.start_thinking()

        # Two-tier routing: 26B stays loaded always, e2b handles simple queries.
        # With OLLAMA_MAX_LOADED_MODELS=2, both coexist — no swap penalty.
        # Only DE-escalate (26B→e2b for speed), never escalate (avoids 3min swap).
        self._maybe_use_fast_model(user_text)

        self._adapt_to_prompt(user_text)
        self._apply_cache_policy()
        self.session.messages = compact_messages(self.session.messages, max_chars=max(8000, self._effective_context_chars() * 2))

        # -- Context gathering — skip heavy stuff for simple creation tasks --
        ctx_chars = self._effective_context_chars()
        context_result = ""
        retrieval_result = ""
        cartridge_result = ""
        skill_result = ""
        plan_result = None
        draft_result = ""

        try:
            # Always gather context — model needs to know what files exist
            with ThreadPoolExecutor(max_workers=5) as pool:
                    futures = {
                        pool.submit(build_context_block, self.repo_root, self.session.pinned_files, ctx_chars): "context",
                        pool.submit(self._retrieval_context, user_text): "retrieval",
                        pool.submit(build_repo_cartridge, self.repo_root, user_text, self.profile.retrieval_budget): "cartridge",
                        pool.submit(resolve_referenced_skills, self.repo_root, user_text): "skills",
                        pool.submit(self.plan_for_task, user_text): "plan",
                    }
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            result = future.result()
                            if key == "context":
                                context_result = result
                            elif key == "retrieval":
                                retrieval_result = result
                            elif key == "cartridge":
                                cartridge_result = result
                            elif key == "skills":
                                skill_result = "\n\n".join(f"Skill {name}:\n{content}" for name, content in result)
                            elif key == "plan":
                                plan_result = result
                        except Exception:
                            pass
        except KeyboardInterrupt:
            self.out.done()
            self.console.print("\n  [dim]interrupted[/]")
            return ""

        context = context_result
        if retrieval_result:
            context = f"{context}\n\nIndexed code matches:\n{retrieval_result}"
        if cartridge_result:
            context = f"{context}\n\nRepo cartridge:\n{cartridge_result}"

        # No system prompt here — agent_loop sets task-specific ones
        system_prompt = ""

        # Handle audio: for HF/MLX pass natively, for Ollama transcribe first
        effective_text = user_text
        if audio:
            if self.config.runtime.provider in ("huggingface-local", "mlx-local"):
                # Native audio — will be part of multipart message
                self.store.append_event(self.session, "audio", f"{len(audio)} clip(s)")
            else:
                # Ollama doesn't support native audio — transcribe and prepend
                for aud in audio:
                    transcript = audio_to_text_fallback(aud, self.config)
                    effective_text = f"[Audio transcription from {aud.source} ({aud.duration_str})]: {transcript}\n\n{effective_text}"
                    self.store.append_event(self.session, "audio_transcribe", f"{aud.source} -> {len(transcript)} chars")

        # Pass image base64 data if any
        image_b64_list = [img.base64_data for img in (images or [])] or None
        composed_messages = compose_messages(
            self.profile,
            system_prompt,
            context,
            self.session.messages,
            effective_text,
            images=image_b64_list,
            provider=self.config.runtime.provider,
        )

        # For HF/MLX with native audio, inject audio content into the last user message
        if audio and self.config.runtime.provider in ("huggingface-local", "mlx-local"):
            last_msg = composed_messages[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content", "")
                if isinstance(content, str):
                    content_parts = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    content_parts = list(content)
                else:
                    content_parts = [{"type": "text", "text": str(content)}]
                for aud in audio:
                    content_parts.append({"type": "audio", "audio": aud.source})
                composed_messages[-1] = {"role": "user", "content": content_parts}

        # Log attachments
        if images:
            self.store.append_event(self.session, "images", f"{len(images)} image(s), {sum(i.size_kb for i in images)}KB")

        # Lifecycle hook: user prompt submit
        hook_result = self.hooks.on_user_prompt_submit(user_text)
        if hook_result.blocked:
            self.console.print(f"[dim]  Hook blocked prompt: {hook_result.error or hook_result.output}[/]")
            return
        # Start turn tracking
        self.turn_tracker.start_turn(watched_files=self.session.pinned_files)
        _turn_start = time.time()

        self.session.messages.append({"role": "user", "content": user_text})
        self.store.append_event(self.session, "user", user_text[:160])
        self.history.record_user_prompt(
            self.session.session_id, str(self.repo_root),
            user_text, model=self.runtime_model,
        )

        changes_before = self.toolkit.changes.change_count
        assistant_text = ""

        # Pass the ask-level indicator to model methods (ONE indicator for entire flow)
        out = self.out

        try:
            # State machine agent loop: harness controls sequencing, model generates content.
            # Based on Codex/Claude analysis: GATHER → WRITE → VERIFY → FIX → DONE
            from .agent import run_agent_loop
            assistant_text = run_agent_loop(
                self, user_text, composed_messages, self.out,
            )
        except KeyboardInterrupt:
            self.out.done()
            # Auto-save ghost snapshot on interrupt
            create_snapshot(
                self.session.session_id, str(self.repo_root),
                self.session.messages, self.session.pinned_files,
                model=self.runtime_model, reason="interrupt",
                capture_files=self.session.pinned_files,
            )
            self.console.print("\n[dim]  Snapshot saved. Use /snapshot list to see saved points.[/]")
            return ""
        except RuntimeErrorWithContext as exc:
            lowered_exc = str(exc).lower()
            retried = False
            if any(token in lowered_exc for token in ("context", "prompt too long", "token", "overflow")):
                max_ctx = self._effective_context_chars()
                compacted, was_compacted = compact_if_needed(self.session.messages, max_ctx, keep_recent=3)
                if was_compacted:
                    self.session.messages = compacted
                    self.store.append_event(self.session, "exec:reactive_compact", "Compacted context after runtime overflow", error=str(exc)[:200])
                    self.store.save(self.session)
                    try:
                        from .agent import run_agent_loop
                        assistant_text = run_agent_loop(self, user_text, composed_messages, self.out)
                        retried = True
                    except Exception:
                        retried = False
            if not retried:
                self.out.set_error(f"Connection issue: {exc}")
                self.log.exception("Runtime error")
                try:
                    simple = [{"role": "user", "content": user_text}]
                    response = self.engine.chat_once(simple)
                    assistant_text = response.get("message", {}).get("content", "")
                    if stream and assistant_text:
                        self.console.print(f"\n{assistant_text}\n")
                except Exception:
                    self.console.print(f"  [red]Failed: {exc}[/]")
                return ""
        except Exception as exc:
            self.out.set_error(str(exc))
            self.log.exception("Chat error")
            return ""

        # Done — clean up output
        self.out.done()

        self.session.last_assistant_text = assistant_text
        # Don't pollute history with garbage — model will mirror it
        is_garbage = (
            not assistant_text.strip()
            or "<unused" in assistant_text
            or len(assistant_text) > 50 and " " not in assistant_text[:100]
        )
        if not is_garbage:
            self.session.messages.append({"role": "assistant", "content": assistant_text})
        self.store.append_event(self.session, "assistant", assistant_text[:160])
        self.store.save(self.session)
        # Turn diff summary
        turn_diff = self.turn_tracker.end_turn()
        if turn_diff.changes:
            print_turn_diff(turn_diff)
        # Desktop notification for slow tasks
        notify_if_slow("gem", f"Task complete: {user_text[:40]}", _turn_start)
        # Record in unified history
        changed_files = self.toolkit.changes.recent_files(since=changes_before)
        self.history.record_assistant_response(
            self.session.session_id, str(self.repo_root),
            assistant_text, model=self.runtime_model,
            files_changed=changed_files if changed_files else [],
        )

        # Show file changes + context budget
        changes_after = self.toolkit.changes.change_count
        if changes_after > changes_before:
            diff = changes_after - changes_before
            self.console.print(f"  [dim]{diff} file(s) changed. Use /verify to run tests, /undo to revert.[/]")

        # Auto-compact if approaching context limit
        max_ctx = self._effective_context_chars()
        self.session.messages, was_compacted = compact_if_needed(self.session.messages, max_ctx)
        if was_compacted:
            self.console.print("[dim]  Context compacted — older messages summarized.[/]")
            self.store.save(self.session)
            # Clear server-side KV cache so stale tokens don't persist
            try:
                import httpx
                httpx.post(f"{self.config.runtime.base_url.rstrip('/')}/v1/chat/completions",
                          json={"model": self.config.runtime.model, "messages": [], "max_tokens": 1,
                                "cache_prompt": False}, timeout=5)
            except Exception:
                pass

        return assistant_text

    # Animated gem icon — pulsates during thinking
    _GEM_ICONS = ["[dim green].[/]", "[green]·[/]", "[bright_green]◆[/]", "[bold bright_green]◆[/]", "[bright_green]◆[/]", "[green]·[/]"]
    _gem_tick = 0
    _token_count = 0

    def _gem_icon(self) -> str:
        self._gem_tick += 1
        return self._GEM_ICONS[self._gem_tick % len(self._GEM_ICONS)]

    def _start_status(self) -> None:
        """Pulsating status with typewriter thinking — reveals text gradually."""
        import threading, time as _time, sys as _sys, os as _os
        self._status_running = True
        self._status_start = _time.time()
        self._thinking_buffer = ""  # full thinking text received so far
        self._thinking_revealed = 0  # how many chars we've shown so far
        self._thinking_lock = threading.Lock()

        def _pulse():
            icons = [".", "·", "◆", "◆", "·", "."]
            tick = 0
            lines_printed = 0  # how many full lines we've already printed
            chars_shown_on_printed_lines = 0
            try:
                cols = _os.get_terminal_size().columns
            except Exception:
                cols = 70

            prefix_len = 10  # " ◆ (10s) " roughly
            indent = "    "  # 4 spaces for wrapped lines
            indent_len = len(indent)
            text_width_first = cols - prefix_len - 2
            text_width_wrap = cols - indent_len - 1
            if text_width_first < 20:
                text_width_first = 50
            if text_width_wrap < 20:
                text_width_wrap = 60

            while self._status_running:
                tick += 1
                icon = icons[tick % len(icons)]
                elapsed = _time.time() - self._status_start
                if elapsed < 60:
                    t = f"{elapsed:.0f}s"
                else:
                    t = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

                with self._thinking_lock:
                    buf = self._thinking_buffer.replace("\n", " ").strip()
                    chars_revealed = self._thinking_revealed
                    remaining = len(buf) - chars_revealed
                    if remaining > 0:
                        if chars_revealed < 40:
                            speed = 6
                        elif chars_revealed < 150:
                            speed = 3
                        else:
                            speed = 1
                        self._thinking_revealed = min(len(buf), chars_revealed + speed)
                    visible = buf[:self._thinking_revealed]

                # Text after already-printed lines
                current_text = visible[chars_shown_on_printed_lines:]
                tw = text_width_first if lines_printed == 0 else text_width_wrap

                # Check if we need to wrap to a new line
                if len(current_text) > tw:
                    full_line = current_text[:tw]
                    if lines_printed == 0:
                        pre = f"\033[32m {icon} ({t}) \033[0m"
                    else:
                        pre = indent
                    _sys.stderr.write(f"\r{pre}\033[2m{full_line}\033[0m\033[K\n")
                    _sys.stderr.flush()
                    chars_shown_on_printed_lines += tw
                    lines_printed += 1
                    current_text = visible[chars_shown_on_printed_lines:]
                    tw = text_width_wrap

                # Current partial line
                if lines_printed == 0:
                    pre = f"\033[32m {icon} ({t}) \033[0m"
                else:
                    pre = indent
                display = current_text[:tw]
                _sys.stderr.write(f"\r{pre}\033[2m{display}\033[0m\033[K")
                _sys.stderr.flush()
                _time.sleep(0.04)

        self._status_thread = threading.Thread(target=_pulse, daemon=True)
        self._status_thread.start()

    def _feed_thinking(self, chunk: str) -> None:
        """Feed a new thinking chunk — the pulse thread reveals it gradually."""
        with self._thinking_lock:
            self._thinking_buffer += chunk

    def _stop_status(self) -> None:
        """Stop and clear."""
        import sys as _sys
        self._status_running = False
        if hasattr(self, "_status_thread"):
            self._status_thread.join(timeout=1)
        _sys.stderr.write("\r\033[K")
        _sys.stderr.flush()
        self._thinking_buffer = ""
        self._thinking_revealed = 0

    def _run_tool_loop_streaming(self, composed_messages: list[dict], stream: bool = True, indicator: ThinkingIndicator | None = None) -> str:
        """Tool-calling loop with clean display."""
        working_messages = composed_messages

        # Route tools based on user message
        user_text = ""
        for msg in reversed(composed_messages):
            if msg.get("role") == "user":
                user_text = str(msg.get("content", ""))
                break
        routing = route_tools(
            user_text,
            self.toolkit.list_tool_names(),
            conversation_history=self.session.messages[-4:],
            online=is_online(),
        )
        self.logger.log_user_input(user_text, intent=",".join(routing.intents))
        if not routing.tool_names:
            return self._run_stream_simple(composed_messages, stream)
        if stream:
            route_bits = ", ".join(routing.intents[:3]) or "chat"
            tool_bits = ", ".join(routing.tool_names[:3])
            self.out.print_info(f"route: {route_bits}")
            if tool_bits:
                self.out.print_info(f"tool lane: {tool_bits}")

        ctx_size = self._select_task_context(user_text, routing.intents)
        output_budget = self._select_task_output_budget(user_text, routing.intents)

        # ── Speed optimization: speculative pre-execution ──
        # Start running predicted tools NOW while model thinks
        if hasattr(self, '_spec_executor'):
            self._spec_executor.predict_and_prefetch(user_text, routing.tool_names)

        # DIRECT FILE EDIT: if intent is edit and we can identify the file,
        # skip the model tool-call loop entirely. Read file, ask model for
        # updated content, write it. One shot. No multi-round failures.
        # Also trigger if recent conversation was about editing a file
        recent_edit_file = self._get_recent_edit_file()
        is_question = "?" in user_text
        is_edit_intent = ("file_edit" in routing.intents or "file_write" in routing.intents) and not is_question
        is_followup_edit = (
            bool(recent_edit_file)
            and not is_edit_intent
            and not is_question
            and any(w in user_text.lower() for w in [
                "add", "change", "remove", "update", "make",
                "modify", "replace", "delete", "insert",
                "improve", "fix", "refactor", "optimize", "rewrite",
                "just", "dunno", "better", "clean",
            ])
        )
        if (is_edit_intent or is_followup_edit) and self.profile.feature_variant in ("compact", "balanced"):
            if stream and (recent_edit_file if is_followup_edit else True):
                target = recent_edit_file if is_followup_edit else "resolved from request"
                self.out.print_info(f"edit path: direct file edit ({target})")
            result = self._direct_file_edit(
                user_text, stream,
                force_file=recent_edit_file if is_followup_edit else None,
            )
            if result is not None:
                return result

        use_minimal = self.profile.feature_variant == "compact"
        all_schemas = self.toolkit.schemas(minimal=use_minimal)
        tools = [t for t in all_schemas if t["function"]["name"] in routing.tool_names]
        # Use centralized output manager
        out = self.out

        try:
            consecutive_errors = 0
            last_tool_call = ""
            max_rounds = 20  # safety cap — Codex has none but we need one for local models
            for _round in range(max_rounds):
                thinking_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls_found: list[dict] = []
                content_started = False
                round_started = time.time()
                first_token_s: float | None = None

                # ── Speed optimization: disable thinking for tool dispatch rounds ──
                # Round 0: think only for complex tasks (file edits, multi-step)
                # Round 1+: NEVER think — model is just reacting to tool results
                if _round == 0:
                    use_think = len(user_text) > 30 or bool({"file_edit", "file_write"} & set(routing.intents))
                else:
                    use_think = False  # continuation rounds don't need reasoning
                round_budget = output_budget if _round == 0 else min(800, output_budget)
                for event in self.engine.stream_chat_events(
                    working_messages,
                    tools=tools,
                    think=use_think,
                    num_ctx=ctx_size,
                    num_predict=round_budget,
                ):
                    if event["type"] == "thinking":
                        chunk = str(event["content"])
                        thinking_parts.append(chunk)
                        if stream:
                            out.feed_thinking(chunk)
                            peek = " ".join("".join(thinking_parts).split())[-120:]
                            if peek:
                                out.set_thinking_peek(peek)
                    elif event["type"] == "content":
                        chunk = str(event["content"])
                        # Filter special tokens
                        if "<|" in chunk or "|>" in chunk:
                            from .tool_parsing import parse_tool_calls as _ptc
                            parsed = _ptc(chunk)
                            if parsed.has_tools:
                                tool_calls_found = parsed.to_ollama_format()
                                break
                            import re as _re
                            chunk = _re.sub(r'<\|[^>]*\|>', '', chunk)
                        if not chunk.strip():
                            continue
                        if first_token_s is None:
                            first_token_s = time.time() - round_started
                        content_parts.append(chunk)
                        if stream:
                            out.stream(chunk)
                            content_started = True
                    elif event["type"] == "tool_calls":
                        tool_calls_found = event["tool_calls"]
                        break

                thinking = "".join(thinking_parts)
                content = "".join(content_parts).strip()
                self._record_runtime_sample(first_token_s=first_token_s, total_s=time.time() - round_started)
                self.stats.record(self.engine.last_response_meta)

                # Check if content contains a text-format tool call (model output it as text)
                if not tool_calls_found and content:
                    import re as _re
                    # Match: write_file(path='x', content='...') or similar
                    tc_match = _re.search(r'(write_file|read_file|edit_file|bash)\s*\(', content)
                    if tc_match:
                        from .tool_parsing import parse_tool_calls as _ptc
                        parsed = _ptc(content)
                        if parsed.has_tools:
                            tool_calls_found = parsed.to_ollama_format()
                            content = parsed.content  # strip tool call from display

                if not content_started and stream:
                    out.done()

                if tool_calls_found:
                    # Dedup: don't repeat the same tool call
                    call_sig = str([(t.get("function",{}).get("name",""), t.get("function",{}).get("arguments","")) for t in tool_calls_found])
                    if call_sig == last_tool_call:
                        break
                    last_tool_call = call_sig

                    # Permission check + execute each tool
                    tool_messages = []
                    for tc in tool_calls_found:
                        f = tc.get("function", {})
                        name = f.get("name", "")
                        args = f.get("arguments", {})
                        if isinstance(args, str):
                            import json as _json
                            try: args = _json.loads(args)
                            except: args = {}

                        # Lifecycle hook: pre-tool
                        hook = self.hooks.on_pre_tool_use(name, args)
                        if hook.blocked:
                            if stream:
                                out.log_tool(name, f"BLOCKED by hook: {hook.output or hook.error}")
                            tool_messages.append({"role": "tool", "content": f"Blocked by hook: {hook.output or hook.error}"})
                            continue

                        allowed, reason = self.perms.check(name, args)
                        if not allowed:
                            if stream:
                                out.log_tool(name, f"SKIPPED: {reason}")
                            tool_messages.append({"role": "tool", "content": f"Denied: {reason}"})
                            continue

                        if stream:
                            out.log_tool(name, str(args)[:60])
                        # Check speculative cache first — instant if pre-fetched
                        spec_result = self._spec_executor.get_if_ready(name, args)
                        if spec_result is not None:
                            result = {"role": "tool", "content": spec_result}
                        else:
                            result = self.toolkit._execute_one(tc)
                        is_err = result["content"].startswith("Error") or result["content"].startswith("Tool error")
                        if stream:
                            out.tool_result(result["content"][:120], error=is_err)
                        self.logger.log_tool_call(name, args, result["content"], duration_ms=0)
                        self.hooks.on_post_tool_use(name, args, result["content"], is_err)
                        # Track file changes for turn diff
                        if name in ("write_file", "edit_file") and "path" in args:
                            self.turn_tracker.track_file(str(args["path"]))
                        tool_messages.append(result)

                    # Check for errors — if too many consecutive, break and force-edit
                    all_errors = all(
                        m["content"].startswith("Error") or m["content"].startswith("Tool error")
                        for m in tool_messages
                    )
                    if all_errors:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                    if stream:
                        for item in tool_messages:
                            is_err = item["content"].startswith("Error") or item["content"].startswith("Tool error")
                            out.tool_result(item["content"], error=is_err)

                    # If ANY file tool error, do the edit ourselves immediately
                    if consecutive_errors >= 1 and bool({"file_edit", "file_write"} & set(routing.intents)):
                        if stream:
                            ResponseDisplay.print_info("applying edit directly...")
                        # Find the file being edited
                        import re as _re
                        fpath = None
                        for tc in tool_calls_found:
                            args = tc.get("function", {}).get("arguments", {})
                            if "path" in args:
                                fpath = args["path"]
                                break
                        if not fpath:
                            fpath_match = _re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                            fpath = fpath_match.group(1) if fpath_match else None
                        if fpath and (self.repo_root / fpath).is_file():
                            old = (self.repo_root / fpath).read_text(errors="replace")
                            code_r = self.engine.chat_once([
                                {"role": "user", "content": f"Current {fpath}:\n```\n{old}\n```\n\nApply this change: {user_text}\n\nIMPORTANT: Make MINIMAL changes. Keep ALL existing code. Only modify what was requested.\nReturn the COMPLETE file with the small change applied. No explanation."}
                            ])
                            new = code_r.get("message", {}).get("content", "").strip()
                            new = _re.sub(r'^```\w*\n', '', new)
                            new = _re.sub(r'\n```\s*$', '', new)
                            if new and new != old:
                                self.toolkit.changes.snapshot_before(fpath, "direct_edit")
                                (self.repo_root / fpath).write_text(new)
                                if stream:
                                    out.log_tool("write_file", f"path={fpath}")
                                    out.tool_result(f"Edited {fpath}")
                                return f"I've updated {fpath} with the requested changes."
                        break

                    if stream:
                        out.start_thinking()  # restart indicator for next round

                    working_messages = [
                        *working_messages,
                        {"role": "assistant", "content": content, "tool_calls": tool_calls_found},
                        *tool_messages,
                        {"role": "user", "content": f"Good. Now do the NEXT step. Original task: {user_text}"},
                    ]
                    continue

                # Force tool if model didn't call one but should have
                if _round <= 1 and not tool_calls_found:
                    content_lower = content.lower()
                    # Check if file write/edit was expected but not done
                    file_intents = {"file_write", "file_edit"}
                    file_tools_called = any(
                        t.get("function", {}).get("name", "") in ("write_file", "edit_file")
                        for t in tool_calls_found
                    )
                    should_force = (
                        not content.strip()  # empty response
                        or any(p in content_lower for p in [
                            "i do not have access", "i cannot", "i don't have",
                            "i recommend using", "beyond my", "not able to",
                            "i am unable", "i'm unable", "cannot directly",
                            "i will now", "i will edit", "i will update",  # says it will but didn't
                        ])
                        or (bool(file_intents & set(routing.intents)) and not file_tools_called)  # edit/write intent but no tool
                    )
                    if should_force and thinking:
                        thinking_lower = thinking.lower()
                        # Try each tool — match name or name with underscores replaced
                        forced = False
                        for ts in tools:
                            tname = ts["function"]["name"]
                            readable = tname.replace("_", " ")
                            if tname in thinking_lower or readable in thinking_lower or f"`{tname}`" in thinking_lower:
                                forced_args = self._extract_args_from_thinking(tname, thinking, user_text)
                                if forced_args is None:
                                    import re as _re
                                    if tname in ("write_file", "edit_file"):
                                        # e2b can't call file tools — generate code, write ourselves
                                        path_match = _re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                                        fpath = path_match.group(1) if path_match else "output.py"
                                        existing = (self.repo_root / fpath)
                                        if existing.is_file() and tname == "edit_file":
                                            # Edit: ask model for the UPDATED version of the file
                                            old_content = existing.read_text(errors="replace")
                                            if stream:
                                                ResponseDisplay.print_info(f"generating edit for {fpath}...")
                                            code_r = self.engine.chat_once([
                                                {"role": "user", "content": f"Here is the current content of {fpath}:\n```\n{old_content}\n```\n\nModify it to: {user_text}\n\nReturn ONLY the complete updated file. No explanation."}
                                            ])
                                        else:
                                            if stream:
                                                ResponseDisplay.print_info(f"generating code for {fpath}...")
                                            code_r = self.engine.chat_once([
                                                {"role": "user", "content": f"Write ONLY the code for {fpath}. No explanation, just the code. Task: {user_text}"}
                                            ])
                                        code_content = code_r.get("message", {}).get("content", "").strip()
                                        code_content = _re.sub(r'^```\w*\n', '', code_content)
                                        code_content = _re.sub(r'\n```\s*$', '', code_content)
                                        if code_content:
                                            forced_args = {"path": fpath, "content": code_content}
                                            forced_calls = [{"function": {"name": "write_file", "arguments": forced_args}}]
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    forced_calls = [{"function": {"name": tname, "arguments": forced_args}}]
                                tool_msgs = self.toolkit.execute_tool_calls(forced_calls)
                                if stream:
                                    out.log_tool(tname, str(forced_args)[:60])
                                    for item in tool_msgs:
                                        out.tool_result(item["content"])
                                    out.start_thinking()  # restart indicator for next round
                                working_messages = [
                                    *working_messages,
                                    {"role": "assistant", "content": "", "tool_calls": forced_calls},
                                    *tool_msgs,
                                    {"role": "user", "content": f"Good. Now do the NEXT step. Original task: {user_text}"},
                                ]
                                forced = True
                                break
                        if forced and len(working_messages) > len(composed_messages):
                            continue
                        # Last resort: retry with ultra-short prompt (no system context)
                        if not forced and not content.strip():
                            retry_msgs = [
                                {"role": "system", "content": "Use tools. Always call tools."},
                                {"role": "user", "content": user_text},
                            ]
                            r = self.engine.chat_once(retry_msgs, tools=tools)
                            tc = r.get("message", {}).get("tool_calls", [])
                            if tc:
                                tool_msgs = self.toolkit.execute_tool_calls(tc)
                                if stream:
                                    for t in tc:
                                        f = t.get("function", {})
                                        out.log_tool(f.get("name", ""), str(f.get("arguments", ""))[:60])
                                    for item in tool_msgs:
                                        out.tool_result(item["content"])
                                    out.start_thinking()  # restart indicator for next round
                                working_messages = [
                                    *working_messages,
                                    r["message"],
                                    *tool_msgs,
                                    {"role": "user", "content": f"Answer: {user_text}"},
                                ]
                                continue
                            content = r.get("message", {}).get("content", "")

                # Auto-save: if response contains a code block with a filename, write it
                if content and "```" in content:
                    import re as _re
                    # Look for patterns like "save to pong.py" or "file named pong.py" in the text
                    file_hint = _re.search(r'(?:save|file|named|called|create)\s+(?:it\s+(?:as|to)\s+)?[`*]*(\w+\.(?:py|js|ts|html|css|json|md|sh))[`*]*', content, _re.IGNORECASE)
                    if file_hint:
                        fname = file_hint.group(1)
                        # Extract the code block
                        code_match = _re.search(r'```(?:\w+)?\n(.*?)```', content, _re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            if len(code) > 20:
                                fpath = self.repo_root / fname
                                self.toolkit.changes.snapshot_before(fname, "auto_write")
                                fpath.parent.mkdir(parents=True, exist_ok=True)
                                fpath.write_text(code)
                                if stream:
                                    self.console.print(f"\n  [green]✓[/] Auto-saved to {fname} ({len(code.splitlines())} lines)")

                if stream:
                    pass  # output handled by out.done()
                return content

            return ""
        finally:
            out.done()

    def _get_recent_edit_file(self) -> str | None:
        """Check if we recently edited/read a file (for follow-up edits)."""
        for event in reversed(self.session.events[-10:]):
            detail = event.get("detail", "")
            if "Updated " in detail or "read_file" in detail or "write_file" in detail:
                import re
                m = re.search(r'(\w[\w.-]*\.(?:py|js|ts|json|md|txt|html|css))', detail)
                if m:
                    return m.group(1)
        # Also check undo changelog
        if self.toolkit.changes.snapshots:
            return self.toolkit.changes.snapshots[-1].path
        return None

    def _direct_file_edit(self, user_text: str, stream: bool = True, force_file: str | None = None) -> str | None:
        """Directly edit a file without relying on model tool calls.

        For small models that can't reliably call edit_file:
        1. Find the file path in the user's message
        2. Read the file ourselves
        3. Ask the model to generate updated content
        4. Write it
        """
        import re
        out = self.out
        tool_display = ToolCallDisplay()
        response = ResponseDisplay()

        if not force_file and self._is_quality_sensitive_creation(user_text):
            return None

        # Find file path
        fpath = force_file  # use forced file if provided (follow-up edit)

        if not fpath:
            # 1. Try exact filename with extension
            path_match = re.search(r'(\w[\w.-]*\.(?:py|js|ts|json|md|txt|html|css|yaml|toml|sh))', user_text)
            fpath = path_match.group(1) if path_match else None

        if not fpath:
            # 2. Fuzzy match against actual repo files
            # Priority: exact stem > startswith > contains (longer match wins)
            skip = {"edit", "file", "make", "the", "function", "should", "help",
                    "write", "add", "create", "bro", "you", "need", "that", "use",
                    "any", "libraries", "with", "and", "classifies", "images"}
            words = [w for w in re.findall(r'\b(\w{3,})\b', user_text.lower()) if w not in skip]
            repo_files = [f for f in self.repo_root.iterdir() if f.is_file() and not f.name.startswith(".")]

            best_match = None
            best_score = 0
            for word in words:
                for f in repo_files:
                    stem = f.stem.lower()
                    if stem == word:
                        score = 100  # exact match
                    elif stem.startswith(word) or word.startswith(stem):
                        score = 50 + len(word)  # prefix match, longer = better
                    elif word in stem:
                        score = 20 + len(word)
                    else:
                        continue
                    if score > best_score:
                        best_score = score
                        best_match = f.name
            fpath = best_match

        if not fpath:
            return None

        full_path = self.repo_root / fpath
        if not full_path.is_file():
            return None

        # Read the file
        old_content = full_path.read_text(errors="replace")
        if stream:
            out.log_tool("read_file", f"path={fpath}")
            out.tool_result(f"{fpath} ({len(old_content.splitlines())} lines)")

        # If user just said a filename with no action, ask what to do
        # Otherwise, always proceed — let the model figure out the intent
        words_without_filename = re.sub(r'\b\w+\.\w{2,4}\b', '', user_text).strip()
        if len(words_without_filename) < 5:  # basically just a filename
            if stream:
                self.console.print(f"I've read {fpath}. What changes would you like me to make?")
            return ""

        # Ask model to generate updated content
        if stream:
            out.set_stage(f"editing {fpath}")
            out.start_thinking()

        old_max = self.config.runtime.max_context_chars
        try:
            num_lines = len(old_content.splitlines())
            use_patch_format = num_lines > 40 or len(old_content) > 1800

            if not use_patch_format:
                r = self.engine.chat_once([
                    {"role": "user", "content": (
                        f"Here is {fpath}:\n```\n{old_content}\n```\n\n"
                        f"Apply this change: {user_text}\n\n"
                        f"Return the COMPLETE updated file. Keep ALL existing code. No explanation."
                    )}
                ], think=False, num_predict=min(2200, max(600, len(old_content) // 2)))
            else:
                r = self.engine.chat_once([
                    {"role": "user", "content": (
                        f"Here is {fpath} ({num_lines} lines):\n```\n{old_content}\n```\n\n"
                        f"Apply this change: {user_text}\n\n"
                        f"Return ONLY the changes. For each change write:\n"
                        f"SEARCH:\n```\nexact old lines\n```\n"
                        f"REPLACE:\n```\nnew lines\n```\n\n"
                        f"Use 2-6 lines of context in SEARCH to match uniquely. No explanation."
                    )}
                ], think=False, num_predict=1200)

            # Restore
            self.config.runtime.max_context_chars = old_max
        finally:
            if stream:
                out.done()

        raw_response = r.get("message", {}).get("content", "").strip()

        # For large files, parse SEARCH/REPLACE blocks and apply them
        if use_patch_format and ("SEARCH:" in raw_response or "SEARCH\n" in raw_response):
            new_content = old_content
            # Extract search/replace pairs from the response
            search_blocks = re.split(r'SEARCH:\s*\n```[^\n]*\n', raw_response)
            applied = 0
            for block in search_blocks[1:]:
                # Find the search text (between first ``` pair)
                search_end = block.find("\n```")
                if search_end < 0:
                    continue
                search_text = block[:search_end]
                # Find the replace text
                replace_match = re.search(r'REPLACE:\s*\n```[^\n]*\n(.*?)\n```', block, re.DOTALL)
                if replace_match:
                    replace_text = replace_match.group(1)
                    if search_text.strip() and search_text.strip() in new_content:
                        new_content = new_content.replace(search_text.strip(), replace_text.strip(), 1)
                        applied += 1
            if applied == 0:
                # Fallback: treat entire response as complete file
                new_content = re.sub(r'^```\w*\n', '', raw_response)
                new_content = re.sub(r'\n```\s*$', '', new_content).strip()
            elif stream:
                self.console.print(f"  [dim]{applied} edit(s) applied[/]")
        else:
            new_content = raw_response
        # Strip markdown fences
        new_content = re.sub(r'^```\w*\n', '', new_content)
        new_content = re.sub(r'\n```\s*$', '', new_content)
        new_content = re.sub(r'^```$', '', new_content, flags=re.MULTILINE).strip()

        if not new_content or new_content == old_content:
            self.console.print("No changes generated.")
            return ""

        # Write the file
        self.toolkit.changes.snapshot_before(fpath, "direct_edit")
        full_path.write_text(new_content)

        # Show what changed
        import difflib
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{fpath}", tofile=f"b/{fpath}", lineterm=""))

        self.console.print(f"  [green]✓[/] Updated {fpath} ({len(new_lines)} lines)")

        # Show compact diff summary
        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        if added or removed:
            self.console.print(f"  [green]+{added}[/] [red]-{removed}[/] lines changed")
            # Show first few changes
            changes_shown = 0
            for line in diff:
                if changes_shown >= 8:
                    self.console.print(f"  [dim]  ... and more[/]")
                    break
                if line.startswith("+") and not line.startswith("+++"):
                    self.console.print(f"  [green]{line}[/]")
                    changes_shown += 1
                elif line.startswith("-") and not line.startswith("---"):
                    self.console.print(f"  [red]{line}[/]")
                    changes_shown += 1

        # LSP diagnostics — ruff/pyflakes/py_compile
        if fpath.endswith(".py"):
            from .lsp import get_diagnostics
            diags = get_diagnostics(full_path)
            if not diags:
                self.console.print(f"  [green]✓[/] No issues found")
            else:
                for d in diags[:5]:
                    color = "red" if d.severity == "error" else "yellow"
                    self.console.print(f"  [{color}]{d}[/]")

        if diff:
            changed_chunks = max(1, sum(1 for line in diff if line.startswith("@@")))
            self.console.print(f"\n  [dim]{fpath}: {changed_chunks} changed region(s), +{added}/-{removed} lines[/]")

        return f"Updated {fpath}."

    @staticmethod
    def _extract_args_from_thinking(tool_name: str, thinking: str, user_text: str) -> dict | None:
        """Extract tool arguments from thinking. Returns None if can't extract reliably."""
        import re
        if tool_name == "current_datetime":
            return {}
        if tool_name in ("web_search",):
            return {"query": user_text}
        if tool_name == "write_file":
            # Only force write_file if we found ACTUAL code in thinking
            code_match = re.search(r'```(?:\w+)?\n(.*?)```', thinking, re.DOTALL)
            if code_match:
                path_match = re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                path = path_match.group(1) if path_match else "output.py"
                return {"path": path, "content": code_match.group(1).strip()}
            # No code found — DON'T force with garbage, return None to skip
            return None
        if tool_name == "read_file":
            path_match = re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
            return {"path": path_match.group(1)} if path_match else None
        if tool_name == "bash":
            cmd_match = re.search(r'`([^`]+)`', thinking)
            return {"command": cmd_match.group(1)} if cmd_match else None
        if tool_name == "grep":
            return {"pattern": user_text.split()[-1] if user_text.split() else None}
        return None

    def _run_stream_simple(self, composed_messages: list[dict], stream: bool = True) -> str:
        """Simple streaming without tools."""
        chunks: list[str] = []
        started_content = False
        out = self.out
        started_at = time.time()
        first_token_s: float | None = None
        max_tokens = 1200
        if stream:
            out.set_thinking_peek("contacting model and waiting for first tokens")

        try:
            for event in self.engine.stream_chat_events(composed_messages, num_predict=max_tokens):
                if event["type"] == "thinking":
                    if stream:
                        out.feed_thinking(str(event["content"]))
                    continue
                if event["type"] == "content":
                    chunk = str(event["content"])
                    # Filter raw Gemma 4 special tokens
                    if "<|" in chunk or "|>" in chunk:
                        import re as _re
                        chunk = _re.sub(r'<\|[^>]*\|>', '', chunk)
                    if not chunk:
                        continue
                    if first_token_s is None:
                        first_token_s = time.time() - started_at
                        if stream:
                            out.set_thinking_peek("model responded, streaming answer")
                    chunks.append(chunk)
                    if stream:
                        out.stream(chunk)
                        started_content = True
            if stream:
                out.done()
        finally:
            self._record_runtime_sample(first_token_s=first_token_s, total_s=time.time() - started_at)

        return "".join(chunks).strip()

    def _select_task_context(self, user_text: str, intents: list[str]) -> int:
        text = user_text.lower()
        simple_intents = {"time", "chat", "git", "web"}
        if intents and set(intents) <= simple_intents:
            return 4096

        quality_task = self._is_quality_sensitive_creation(text)
        single_file = bool(Path(text.split()[-1]).suffix) or any(ext in text for ext in (".py", ".js", ".ts", ".html", ".css"))
        if quality_task and ("game" in text or "app" in text):
            return 8192 if single_file else 12288

        if set(intents) & {"file_edit", "file_write", "search_code", "quality_create"}:
            return 8192 if single_file else 12288

        return 6144

    def _select_task_output_budget(self, user_text: str, intents: list[str]) -> int:
        text = user_text.lower()
        if set(intents) & {"quality_create"}:
            return 2200
        if set(intents) & {"file_edit"}:
            return 1200
        if set(intents) & {"file_write"}:
            if any(ext in text for ext in (".py", ".js", ".ts", ".html", ".css")):
                return 1600
            return 1200
        if set(intents) & {"search_code", "git"}:
            return 700
        return 900

    def _maybe_auto_verify(self) -> None:
        """Auto-suggest running tests after file changes in interactive mode."""
        verify_cmd = guess_verify_command(self.repo_root)
        if not verify_cmd:
            return
        change_count = self.toolkit.changes.change_count
        self.console.print(
            f"\n  [dim]{change_count} file(s) changed.[/] "
            f"[yellow]Run tests?[/] [dim]({verify_cmd})[/]"
        )
        try:
            answer = self.prompt.prompt("  [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer in {"y", "yes"}:
            self.console.print()
            output, code = run_verification(self.repo_root, bias=self.profile.verification_bias)
            style = "green" if code == 0 else "red"
            self.console.print(Panel(
                output[-4000:],
                title=f"[{style}]verification exit={code}[/]",
                border_style=style,
            ))
            self.store.append_event(self.session, "auto_verify", f"exit={code}")

    def _render_thinking(self, thinking_text: str) -> None:
        mode = self.config.ui.thinking_mode
        if mode == "hidden" or not thinking_text:
            return
        self._thinking_tick += 1
        frame, label = thinking_frame(self._thinking_tick)
        if mode == "summary":
            one_line = " ".join(thinking_text.split())
            preview = one_line[:240] + ("..." if len(one_line) > 240 else "")
            body = f"  {frame}  [dim]{label}[/]\n  [dim]{preview}[/]"
        else:
            # full mode
            body = f"  {frame}  [dim]{label}[/]\n  {thinking_text[-1800:]}"
        self.console.print(Panel(body, title="[dim]thinking[/]", border_style="dim"))

    def _approve_tool_calls(self, summaries: list[str]) -> bool:
        tool_names = [summary.split("(", 1)[0] for summary in summaries]
        decisions = [self._permission_decision(name) for name in tool_names]
        if decisions and all(decision == "allow" for decision in decisions):
            return True
        if decisions and any(decision == "deny" for decision in decisions):
            return False
        decision = self.approvals.review(
            "Tool Calls",
            [ApprovalItem(label=name, detail=summary) for name, summary in zip(tool_names, summaries)],
            allow_repo_option=True,
        )
        if decision == "session-allow":
            self.session_allows.update(tool_names)
            return True
        if decision == "repo-allow":
            for name in tool_names:
                self.permissions.allow(name)
            return True
        return decision == "allow"

    def _review_patch(self, diff_text: str) -> str | None:
        files = parse_diff(diff_text)
        if not files:
            return None
        selected_files = []
        for file in files:
            file_panel = Panel(
                "\n".join(
                    [f"{hunk.header}\n" + "\n".join(hunk.lines[:20]) + ("\n..." if len(hunk.lines) > 20 else "") for hunk in file.hunks]
                )[:3000],
                title=f"{file.new_path} ({len(file.hunks)} hunks)",
            )
            self.console.print(file_panel)
            mode = self.prompt.prompt("Apply [a]ll hunks, [s]elect hunks, or [n]one? ").strip().lower()
            if mode in {"n", "none"}:
                continue
            if mode in {"s", "select"}:
                chosen_hunks = []
                for idx, hunk in enumerate(file.hunks, start=1):
                    preview = hunk.header + "\n" + "\n".join(hunk.lines[:16])
                    self.console.print(Panel(preview, title=f"{file.new_path} hunk {idx}"))
                    if self._confirm("Include this hunk?"):
                        chosen_hunks.append(hunk)
                if chosen_hunks:
                    file.hunks = chosen_hunks
                    selected_files.append(file)
                continue
            selected_files.append(file)
        if not selected_files:
            return None
        reviewed = build_diff(selected_files)
        preview = reviewed[:1200] + ("..." if len(reviewed) > 1200 else "")
        self.console.print(Panel(preview, title="Selected Patch"))
        return reviewed if self._confirm("Keep this reviewed patch?") else None

    def _retrieval_context(self, query: str) -> str:
        if not query.strip():
            return ""
        if load_index(self.repo_root) is None:
            try:
                count, _ = build_index(self.repo_root)
                self.store.append_event(self.session, "index_build", f"{count} files")
            except Exception:
                return ""
        results = search_index(self.repo_root, query, limit=self.profile.retrieval_budget)
        if not results:
            return ""
        lines = [f"{item['path']}#chunk{item['chunk_id']}: {item['preview']}" for item in results]
        return "\n\n".join(lines)

    def plan_for_task(self, task: str):
        note = build_plan_note(task)
        if not self.config.runtime.planner_enabled:
            return note
        if not self.config.runtime.planner_model or self.config.runtime.planner_model == self.runtime_model:
            return note
        try:
            planner_runtime = replace(
                self.config.runtime,
                model=self.config.runtime.planner_model,
                temperature=0.0,
                max_context_chars=min(8000, self.config.runtime.max_context_chars),
            )
            planner = GemRuntimeGateway(planner_runtime)
            response = planner.chat_once(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a tiny planning lane for Gem. "
                            "Return one short action plan for the coding assistant. "
                            "Keep it under 80 words and focus on the next efficient steps."
                        ),
                    },
                    {"role": "user", "content": task},
                ],
            )
            planner_text = response.get("message", {}).get("content", "").strip()
            if planner_text:
                note.summary = f"{note.summary} Planner: {planner_text[:220]}"
        except Exception:
            pass
        return note

    def planner_checkpoint_hint(self, checkpoint: str, task: str, context_snippet: str = "") -> PlannerHint | None:
        if not self.config.runtime.planner_hints_enabled:
            return None
        if self.profile.key == "gemma4-e2b":
            return None
        compact_context = context_snippet.strip()[:1200]
        prompt = (
            "Return strict JSON only with keys: next_action, likely_file, risk, quality_gap, stop.\n"
            "Be terse. Think like a coding assistant control loop, not a chat assistant.\n"
            "Only suggest the next high-value action.\n\n"
            f"Checkpoint: {checkpoint}\n"
            f"Task: {task}\n"
        )
        if compact_context:
            prompt += f"Context:\n{compact_context}\n"
        try:
            response = self.engine.generate_once(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are Gem's checkpoint planner.\n"
                            "Produce tiny structured execution hints.\n"
                            "No prose. JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=220,
            )
            hint = parse_planner_hint(response, checkpoint)
            if not hint:
                return None
            self.planner_hints.record(hint)
            self.store.append_event(
                self.session,
                "planner_hint",
                f"{checkpoint}: {hint.next_action or hint.risk or hint.quality_gap or 'hint'}",
                likely_file=hint.likely_file,
                risk=hint.risk,
                quality_gap=hint.quality_gap,
                stop=str(hint.stop).lower(),
            )
            return hint
        except Exception:
            return None

    def maybe_escalate_for_task(self, task: str) -> None:
        if not self.config.runtime.escalation_enabled:
            return
        note = self.plan_for_task(task)
        if note.complexity != "high":
            return
        if self.profile.key == "gemma4-e2b":
            target = GEMMA_PROFILES["gemma4-e4b"]
        elif self.profile.key == "gemma4-e4b":
            target = GEMMA_PROFILES["gemma4-26b-moe"]
        else:
            return
        self.profile = target
        self.runtime_model = target.default_model
        self.config.runtime.profile = target.key
        self.config.runtime.model = target.default_model
        self.engine = GemRuntimeGateway(self.config.runtime)
        save_config(self.config)
        self.store.append_event(self.session, "model_escalation", f"{target.key} for {task[:80]}")
        self.render_agent_state("escalation", f"Switched to {target.display_name} for a broader task")

    def _should_orchestrate(self, user_text: str) -> bool:
        """Detect if a task is complex enough to need multi-agent orchestration.

        Auto-triggers orchestrator for tasks that:
        - Involve creating multi-file projects ("make an app", "build a game")
        - Require multiple distinct steps ("set up X, then configure Y, then...")
        - Are explicitly complex ("full", "complete", "entire", "from scratch")

        Simple edits, questions, and single-file tasks stay in the normal loop.
        """
        text = user_text.lower().strip()

        # Never orchestrate questions or short messages
        if "?" in text or len(text) < 20:
            return False

        # Check for multi-step creation tasks
        creation_words = ("make", "build", "create", "set up", "implement", "develop")
        complex_targets = ("app", "game", "project", "website", "server", "api",
                          "system", "framework", "tool", "cli", "dashboard")
        has_creation = any(w in text for w in creation_words)
        has_complex_target = any(w in text for w in complex_targets)

        # Check for explicit complexity signals
        complexity_signals = ("from scratch", "full", "complete", "entire",
                            "multiple files", "multi-file", "step by step",
                            "and then", "after that", "set up everything")
        has_complexity = any(s in text for s in complexity_signals)

        # Multi-step indicators (commas listing steps, "and" chaining)
        import re
        has_multiple_actions = len(re.findall(r'\b(?:and|then|also|plus)\b', text)) >= 2

        return (has_creation and has_complex_target) or has_complexity or has_multiple_actions or self._is_quality_sensitive_creation(text)

    @staticmethod
    def _is_quality_sensitive_creation(user_text: str) -> bool:
        text = user_text.lower().strip()
        creation_words = ("make", "build", "create", "generate", "implement")
        quality_words = (
            "app", "game", "website", "dashboard", "ui", "clone",
            "polish", "authentic", "look like", "feel like", "sonic",
            "beautiful", "high quality", "fidelity", "playable",
        )
        return any(w in text for w in creation_words) and any(w in text for w in quality_words)

    def _generate_task_name(self, user_text: str) -> str:
        """Generate a short dynamic task name from user input (like Codex)."""
        import re
        text = user_text.strip().lower()
        # Quick keyword-based task naming — no LLM call needed
        if any(w in text for w in ("fix", "bug", "error", "broken")):
            f = re.search(r'(\w+\.\w{2,4})', text)
            return f"fixing {f.group(1)}" if f else "fixing bug"
        if any(w in text for w in ("make", "create", "build", "write", "generate")):
            # Try to extract what they're making
            for pattern in [r'(?:make|create|build|write)\s+(?:a\s+)?(.{3,25}?)(?:\s+that|\s+which|\s+for|$)',
                           r'(?:make|create|build)\s+(.{3,20}?)\.?\s*$']:
                m = re.search(pattern, text)
                if m:
                    return f"creating {m.group(1).strip()}"
            return "creating file"
        if any(w in text for w in ("edit", "change", "update", "modify", "refactor")):
            f = re.search(r'(\w+\.\w{2,4})', text)
            return f"editing {f.group(1)}" if f else "editing code"
        if any(w in text for w in ("search", "find", "look", "where")):
            return "searching codebase"
        if any(w in text for w in ("test", "run", "execute")):
            return "running tests"
        if any(w in text for w in ("explain", "what", "how", "why")):
            return "analyzing"
        if any(w in text for w in ("git", "commit", "push", "branch")):
            return "git operation"
        if len(text) > 60:
            return text[:40].rsplit(" ", 1)[0] + "..."
        return text[:40] if len(text) > 5 else ""

    def _maybe_use_fast_model(self, user_text: str) -> None:
        """Use e2b for trivial queries (time, hi, git status) — 3x faster.

        26B stays loaded always. With OLLAMA_MAX_LOADED_MODELS=2, e2b loads
        alongside it without evicting the 26B. After the simple query,
        next complex query goes straight back to the already-loaded 26B.
        """
        if not self.config.runtime.escalation_enabled:
            return

        text = user_text.lower().strip()
        # Only use fast model for truly trivial stuff
        is_trivial = (
            len(text) < 15
            or text in ("hi", "hello", "hey", "thanks", "ok", "yes", "no")
            or any(text.startswith(w) for w in ("what time", "what day", "what date"))
        )

        if is_trivial and "26b" in self.runtime_model.lower():
            draft = self.config.runtime.draft_model
            if draft:
                self._saved_main_model = self.runtime_model
                self.config.runtime.model = draft
                from .runtime import GemRuntimeGateway as _GRG
                self.engine = _GRG(self.config.runtime)
                self.runtime_model = draft
        elif not is_trivial and hasattr(self, '_saved_main_model'):
            # Switch back to 26B (already loaded, instant)
            self.config.runtime.model = self._saved_main_model
            from .runtime import GemRuntimeGateway as _GRG
            self.engine = _GRG(self.config.runtime)
            self.runtime_model = self._saved_main_model
            del self._saved_main_model

    def _maybe_switch_tier(self, user_text: str) -> None:
        """Two-tier model routing: fast model for simple tasks, smart for complex.

        e4b (28 tok/s) handles: simple questions, time, git status, web search
        26B (8 tok/s) handles: code generation, file editing, multi-step tasks
        """
        if not self.config.runtime.escalation_enabled:
            return

        # Check what models are available
        try:
            available = self.engine.list_models()
        except Exception:
            return

        has_26b = any("26b" in m.lower() or "gemma26b" in m.lower() for m in available)
        has_e4b = any("e4b" in m.lower() for m in available)

        if not (has_26b and has_e4b):
            return  # Can't do two-tier without both models

        # Determine complexity from intent
        routing = route_tools(
            user_text,
            self.toolkit.list_tool_names(),
            online=is_online(),
        )

        simple_intents = {"time", "chat", "git", "web", "general"}
        complex_intents = {"file_edit", "file_write", "search_code"}

        is_complex = bool(set(routing.intents) & complex_intents) or len(user_text) > 200

        # Pick the right model
        if is_complex and "e4b" in self.runtime_model:
            # Escalate to 26B for complex tasks
            target_model = next((m for m in available if "26b" in m.lower() or "gemma26b" in m.lower()), None)
            if target_model:
                self.out.print_info(f"↑ escalating to {target_model}")
                self.config.runtime.model = target_model
                self.engine = GemRuntimeGateway(self.config.runtime)
                self.runtime_model = target_model
        elif not is_complex and "26b" in self.runtime_model.lower():
            # De-escalate to e4b for simple tasks (3x faster)
            target_model = next((m for m in available if "e4b" in m.lower()), None)
            if target_model:
                self.out.print_info(f"↓ fast mode: {target_model}")
                self.config.runtime.model = target_model
                self.engine = GemRuntimeGateway(self.config.runtime)
                self.runtime_model = target_model

    def _effective_context_chars(self) -> int:
        base = min(self.config.runtime.max_context_chars, self.profile.recommended_context_chars)
        if self.profile.key == "gemma4-26b-laptop":
            base = min(base, 14000)
        if self.config.runtime.mode == "fast":
            return max(6000, min(base, 12000))
        if self.config.runtime.mode == "deep":
            return max(base, min(self.config.runtime.max_context_chars, base + 12000))
        if self.config.runtime.cache_policy == "rolling":
            cap = 14000 if self.profile.key == "gemma4-26b-laptop" else 16000
            return max(8000, min(base, cap))
        if self.config.runtime.low_overhead_mode:
            return max(6000, min(base, 12000))
        return base

    def _adapt_to_prompt(self, prompt: str) -> None:
        if not self.config.runtime.adaptive_execution:
            return
        lowered = prompt.lower()
        if self.config.runtime.mode == "fast" and any(word in lowered for word in ("large", "architecture", "migration", "multi-file", "refactor")):
            self.config.runtime.max_context_chars = max(self.config.runtime.max_context_chars, 18000)
        if len(prompt) > 3000:
            self.config.runtime.max_context_chars = min(self.config.runtime.max_context_chars, self.profile.recommended_context_chars)

    def _apply_cache_policy(self) -> None:
        window = max(8, self.config.runtime.rolling_window_messages)
        if self.config.runtime.cache_policy == "rolling":
            self.session.messages = self.session.messages[-window:]
            return
        if self.config.runtime.cache_policy == "tight":
            self.session.messages = self.session.messages[-max(8, window // 2):]

    def _draft_assist(self, user_text: str) -> str:
        draft_model = self.config.runtime.draft_model
        if not draft_model or draft_model == self.runtime_model:
            return ""
        try:
            draft_runtime = replace(
                self.config.runtime,
                model=draft_model,
                temperature=0.0,
                max_context_chars=min(6000, self._effective_context_chars()),
            )
            draft_engine = GemRuntimeGateway(draft_runtime)
            response = draft_engine.chat_once(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are Gem's draft lane. "
                            "Return a short draft of the likely answer shape, files, tools, or checks needed. "
                            "Keep it under 120 words."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ]
            )
            text = response.get("message", {}).get("content", "").strip()
            return text[:500]
        except Exception:
            return ""

    def _permission_decision(self, action: str) -> str | None:
        if action in self.session_allows:
            return "allow"
        return self.permissions.decision_for(action)

    def _approve_action(self, action: str, detail: str) -> bool:
        decision = self._permission_decision(action)
        if decision == "allow":
            return True
        if decision == "deny":
            return False
        result = self.approvals.review(
            action.replace("_", " ").title(),
            [ApprovalItem(label=action, detail=detail)],
            allow_repo_option=True,
        )
        if result == "session-allow":
            self.session_allows.add(action)
            return True
        if result == "repo-allow":
            self.permissions.allow(action)
            return True
        return result == "allow"

    def close(self) -> None:
        if hasattr(self, '_ask_indicator'):
            self._ask_out.done()
        self.toolkit.close()
        self.engine.close()
        if self.bg_indexer is not None:
            self.bg_indexer.stop()
        if self.file_watcher is not None:
            self.file_watcher.stop()
        if self.project_watcher is not None:
            self.project_watcher.stop()
