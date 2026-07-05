"""LocalCode Textual TUI — main application."""
from __future__ import annotations

import os
import sys
import threading

# Monkey-patch Textual for Python 3.13: _message_queue sometimes becomes
# a plain list during widget composition. Patch _close_messages and
# post_message to handle this gracefully.
import textual.message_pump as _mp
from textual._queue import Queue as _TextualQueue

_orig_close = _mp.MessagePump._close_messages
async def _safe_close(self, **kwargs):
    if isinstance(self._message_queue, list):
        self._message_queue = _TextualQueue()
    return await _orig_close(self, **kwargs)
_mp.MessagePump._close_messages = _safe_close

_orig_post = _mp.MessagePump.post_message
def _safe_post(self, message):
    if isinstance(self._message_queue, list):
        self._message_queue = _TextualQueue()
    return _orig_post(self, message)
_mp.MessagePump.post_message = _safe_post

from textual.app import App
from textual.binding import Binding

from .bridge import AgentEvent, ApprovalRequest, TUIBridge
from .screens.chat import ChatScreen
from .screens.mode_picker import ModePickerScreen
from .screens.model_picker import ModelPickerScreen
from .screens.setup import SetupScreen


_TERMINAL_RESTORE = (
    "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1004l"
    "\x1b[?1005l\x1b[?1006l\x1b[?1015l\x1b[?2004l"
    "\x1b[?25h\x1b[?1049l\x1b[0m"
)


def _restore_terminal_state() -> None:
    """Best-effort reset for shutdown paths that bypass Textual teardown."""
    payload = _TERMINAL_RESTORE.encode()
    try:
        with open("/dev/tty", "wb", buffering=0) as tty:
            tty.write(payload)
        return
    except Exception:
        pass
    try:
        os.write(1, payload)
    except Exception:
        try:
            sys.__stdout__.write(_TERMINAL_RESTORE)
            sys.__stdout__.flush()
        except Exception:
            pass


class LocalCodeTUI(App):
    """Textual-based terminal UI for LocalCode."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "localcode"

    BINDINGS = [
        Binding("ctrl+c", "copy_or_quit", "Copy/Quit", show=False, priority=True),
    ]

    SCREENS = {
        "chat": ChatScreen,
        "mode_picker": ModePickerScreen,
        "model_picker": ModelPickerScreen,
        "setup": SetupScreen,
    }

    def __init__(self, show_mode_picker: bool = True) -> None:
        # ansi_color must be passed to __init__, not just set as a class
        # attribute — App.__init__ calls set_reactive() with its own
        # default (False), which would otherwise clobber it. With it True,
        # Textual emits raw SGR escapes instead of converting ANSI colors
        # to RGB, so `ansi_default` renders as the terminal's own default
        # background/foreground rather than solid black.
        super().__init__(ansi_color=True)
        # textual-ansi is the built-in theme whose background, surface,
        # panel, foreground and accents all resolve to the terminal's own
        # palette. No custom theme — the terminal decides how it looks.
        if "textual-ansi" in self.available_themes:
            self.theme = "textual-ansi"
        self.show_mode_picker = show_mode_picker
        self.engine = None
        self.config = None
        self.bridge = None
        self._cleaned_up = False
        # Clean-shutdown belt + suspenders. Python's atexit only runs AFTER
        # all non-daemon threads join. When our worker threads are blocked
        # on llama-server I/O, atexit doesn't fire — leaves an orphan
        # llama-server surviving Ctrl+C and getting the 5-minute
        # "executor did not finish joining" warning the user hit.
        #
        # Fix: install SIGINT/SIGTERM handlers that kill llama-server
        # BEFORE Python starts waiting on threads. Then atexit is backup.
        import atexit
        import signal
        atexit.register(self._cleanup)

        def _sig_cleanup(signum, frame):
            try:
                from ..server_manager import ServerManager
                ServerManager.get().shutdown(force=True)
            except Exception:
                pass
            # Stop any in-flight TTS playback (`say` subprocess) — without
            # this the voice keeps talking after the TUI is gone.
            try:
                from ..voice import stop_speaking as _stop
                _stop()
            except Exception:
                pass
            try:
                from ..tools.bash import reap_background_processes
                reap_background_processes()
            except Exception:
                pass
            _restore_terminal_state()
            signal.signal(signum, signal.SIG_DFL)
            try:
                import os
                os.kill(os.getpid(), signum)
            except Exception:
                pass

        # Only install if we're on the main thread (signal module requires it)
        try:
            signal.signal(signal.SIGINT, _sig_cleanup)
            signal.signal(signal.SIGTERM, _sig_cleanup)
        except (ValueError, OSError):
            # Non-main thread or unsupported platform — fall back to atexit.
            pass

    def _cleanup(self) -> None:
        """Ensure backend is fully stopped — called by atexit and action_quit."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        # Kill llama-server FIRST. Any thread blocked on its I/O unblocks
        # immediately, which lets Python actually reach the executor-join.
        try:
            from ..server_manager import ServerManager
            # force=True — app is exiting, kernel reclaims Metal on
            # process death so the 5-10 s graceful dealloc is wasted time.
            ServerManager.get().shutdown(force=True)
        except Exception:
            pass
        # Same for any in-flight TTS — kill it now so the assistant
        # voice doesn't keep talking after the user closed the TUI.
        try:
            from ..voice import stop_speaking as _stop
            _stop()
        except Exception:
            pass
        # Tear down MCP server subprocesses so they don't outlive us.
        try:
            from ..mcp import shutdown_all as _mcp_down
            _mcp_down()
        except Exception:
            pass
        # Reap any test/server apps the agent backgrounded via the bash
        # tool (e.g. `python3 app.py &` to verify a build). Without this,
        # those interpreters spin at 100 % CPU forever after the user
        # quits the TUI.
        try:
            from ..tools.bash import reap_background_processes
            reap_background_processes()
        except Exception:
            pass
        if self.engine is not None:
            try:
                self.engine.out._indicator_running = False
                self.engine.out._stop_indicator()
                self.engine.close()
            except Exception:
                pass

    def on_mount(self) -> None:
        """Initialize config and show first screen.

        Preflight: detect stuck llama-servers or memory pressure. If
        stuck procs are found, auto-run recovery (ONE macOS admin
        dialog, then runs silently). User never sees the terms
        'D-state', 'vm_fault', or commands like 'localcode unstick'.
        """
        # Preview-screen short-circuit (set by `localcode --preview-
        # screen <name>` in the console-script entrypoint). Skips health check, server
        # lifecycle, model-picker decision logic — pushes the
        # requested screen with the minimum mock state it needs to
        # render. Useful for iterating UI tweaks without going through
        # the full new-user flow each time.
        preview = getattr(self, "_preview_screen", None)
        if preview:
            from ..config import load_config
            try:
                self.config = load_config()
            except Exception:
                # Even if config load fails (truly fresh install),
                # build a minimal default so the screen has SOMETHING
                # to read.
                from ..config import AppConfig, RuntimeConfig, SearchConfig, UIConfig, SafetyConfig, LoggingConfig
                self.config = AppConfig(
                    runtime=RuntimeConfig(),
                    search=SearchConfig(),
                    ui=UIConfig(),
                    safety=SafetyConfig(),
                    logging=LoggingConfig(),
                )
            self.bridge = TUIBridge(self)
            screen_map = {
                "setup": "setup",
                "mode_picker": "mode_picker",
                "model_picker": "model_picker",
                "chat": "chat",
            }
            target = screen_map.get(preview, "chat")
            self.push_screen(target)
            return

        from ..config import load_config
        from ..health import check_system_health

        health = check_system_health()
        if not health.ok:
            # Two failure modes: stuck procs (auto-recoverable) vs
            # general pressure (user needs to close apps or reboot).
            if health.stuck_servers:
                import sys
                sys.stderr.write(
                    "\n  Previous session left stuck processes. "
                    "Attempting automatic recovery…\n\n"
                )
                sys.stderr.flush()
                from ..recovery import attempt_recovery
                ok, msg = attempt_recovery(
                    verbose=True,
                    on_progress=lambda m: sys.stderr.write(f"  {m}\n"),
                )
                sys.stderr.write(f"\n  {msg}\n\n")
                sys.stderr.flush()
                if ok:
                    # Recovery succeeded — re-run health check and
                    # continue if it now passes.
                    health = check_system_health()
                    if health.ok:
                        # Fall through to normal init
                        pass
                    else:
                        self.exit(message=health.message)
                        return
                else:
                    self.exit(message=msg)
                    return
            else:
                # Memory pressure is transient on macOS and the runtime/setup
                # path can recover or show a precise model-launch error. Do not
                # make `localcode` fail before rendering the TUI because a
                # preflight snapshot saw low allocatable memory.
                try:
                    import logging
                    logging.getLogger(__name__).warning(
                        "startup health warning ignored: %s", health.message
                    )
                except Exception:
                    pass

        self.config = load_config()
        self.bridge = TUIBridge(self)

        # Always show the model picker on launch — never silently auto-load the
        # last model. The picker highlights the currently-configured model, so a
        # returning user can keep it, while a new user gets to CHOOSE before any
        # multi-GB download begins. Mid-session, /model switches models too.
        def _after_pick(choice) -> None:
            if choice is None:
                self.exit()
                return
            # Persist the selection so setup + runtime see it. The file
            # doesn't need to exist yet; setup will download it.
            self.config.runtime.model = str(choice.local_path)
            try:
                from ..config import save_config
                save_config(self.config)
            except Exception:
                pass
            self.push_screen("setup")

        self.push_screen("model_picker", _after_pick)

    def ensure_backend(self) -> bool:
        """Lazily initialize LocalCodeApp backend. Returns True if ready."""
        if self.engine is not None:
            return True
        try:
            from ..app import LocalCodeApp
            self.engine = LocalCodeApp(self.config)
            self.engine.out.set_event_callback(self.bridge.on_event)
            self.engine.out.set_approval_callback(self.bridge.request_approval)
            # `--resume` flag: load prior session messages BEFORE the
            # chat screen starts streaming events. We resolve "last"
            # to the most recently modified session file. Failures here
            # are non-fatal — the user just starts fresh.
            resume_id = getattr(self, "_resume_session_id", None)
            if resume_id:
                self._apply_resume(resume_id)
            return True
        except Exception as e:
            self.notify(f"Backend error: {e}", severity="error")
            return False

    def _apply_resume(self, resume_id: str) -> None:
        """Restore a prior session: load messages + replay into chat log."""
        try:
            from ..session import SessionStore
            store = SessionStore()
            if resume_id == "last":
                files = sorted(
                    store.sessions_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not files:
                    self.notify("No previous sessions to resume.", severity="warning")
                    return
                resume_id = files[0].stem
            session = store.load(resume_id)
        except Exception as e:
            self.notify(f"Couldn't resume {resume_id}: {e}", severity="error")
            return
        # Hand the loaded messages to the engine + emit a hydrate event
        # so the chat-screen renders them on mount.
        try:
            self.engine.session = session
        except Exception:
            pass
        try:
            self._pending_resume_messages = list(session.messages or [])
        except Exception:
            self._pending_resume_messages = []

    def action_copy_or_quit(self) -> None:
        """Ctrl+C: double-press to exit.

        First press does NOT quit — it cancels any in-flight turn and
        shows a grey "Press Ctrl+C again to exit" hint at the bottom.
        A second press within 3 s actually quits; otherwise the window
        lapses and the next Ctrl+C is treated as a fresh first press.
        This stops an accidental single Ctrl+C from killing a session
        (a long-standing papercut).

        (Selection-copy on Ctrl+C was removed earlier — chat_log's
        on_mouse_up already copies via OSC 52 the moment a drag ends.)
        """
        import time as _t
        now = _t.monotonic()
        if now - getattr(self, "_ctrl_c_at", 0.0) <= 3.0:
            self.action_quit()
            return
        self._ctrl_c_at = now
        # First press also interrupts an in-flight generation (no-op if idle).
        try:
            if getattr(self, "engine", None) is not None:
                self.engine.cancel_requested = True
        except Exception:
            pass
        screen = self.screen
        _hint = getattr(screen, "flash_exit_hint", None)
        if callable(_hint):
            _hint()

    # Route bridge messages to the active screen
    def on_agent_event(self, event: AgentEvent) -> None:
        screen = self.screen
        if hasattr(screen, "on_agent_event"):
            screen.on_agent_event(event)

    def on_approval_request(self, event: ApprovalRequest) -> None:
        screen = self.screen
        if hasattr(screen, "on_approval_request"):
            screen.on_approval_request(event)

    def action_quit(self) -> None:
        """Clean up ALL backend threads and processes before exiting."""
        _restore_terminal_state()
        # Suppress stdout/stderr FIRST — before cancelling workers
        # so any output from dying worker threads goes to devnull
        devnull = open(os.devnull, "w")
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            try:
                if self.engine is not None:
                    self.engine.cancel_requested = True
            except Exception:
                pass
            self.workers.cancel_all()
            self._cleanup()
        except Exception:
            pass
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            devnull.close()
            _restore_terminal_state()
        self.exit()

        def _force_exit() -> None:
            try:
                _restore_terminal_state()
            finally:
                os._exit(0)

        threading.Timer(0.05, _force_exit).start()


def main() -> None:
    """Entry point for lc-tui or localcode --tui."""
    app = LocalCodeTUI(show_mode_picker=False)
    # Mouse events are required for the chat_log's click-to-expand
    # thinking blocks and its custom drag-select + OSC 52 clipboard.
    # Disabling mouse trades terminal-native text selection for both
    # of those, which is a worse UX. The custom selection path
    # already covers what the native one would have done.
    app.run()


if __name__ == "__main__":
    main()
