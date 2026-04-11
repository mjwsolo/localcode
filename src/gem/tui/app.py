"""LocalCode Textual TUI — main application."""
from __future__ import annotations

import os
import sys

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
from .screens.setup import SetupScreen


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
        "setup": SetupScreen,
    }

    def __init__(self, show_mode_picker: bool = True) -> None:
        super().__init__()
        self.show_mode_picker = show_mode_picker
        self.gem_app = None
        self.gem_config = None
        self.bridge = None
        self._cleaned_up = False
        # Register atexit handler for Ctrl+C / terminal kill
        import atexit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Ensure backend is fully stopped — called by atexit and action_quit."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.gem_app is not None:
            try:
                self.gem_app.out._indicator_running = False
                self.gem_app.out._stop_indicator()
                self.gem_app.close()
            except Exception:
                pass

    def on_mount(self) -> None:
        """Initialize config and show first screen. GemApp loaded lazily."""
        from ..config import load_config

        self.gem_config = load_config()
        self.bridge = TUIBridge(self)

        # Check if server is reachable
        from ..runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(self.gem_config.runtime)
        ok, _ = gw.healthcheck()
        if ok:
            # Server running — go straight to chat or mode picker
            saved_mode = getattr(self.gem_config.runtime, 'laptop_26b_runtime_mode', '')
            if saved_mode:
                self.push_screen("chat")
            else:
                self.push_screen("mode_picker")
            return

        # Server not running — always go to setup screen.
        # Setup handles everything: binary check, model download, server launch.
        self.push_screen("setup")

    def ensure_backend(self) -> bool:
        """Lazily initialize GemApp backend. Returns True if ready."""
        if self.gem_app is not None:
            return True
        try:
            from ..app import GemApp
            self.gem_app = GemApp(self.gem_config)
            self.gem_app.out.set_event_callback(self.bridge.on_event)
            self.gem_app.out.set_approval_callback(self.bridge.request_approval)
            return True
        except Exception as e:
            self.notify(f"Backend error: {e}", severity="error")
            return False

    def action_copy_or_quit(self) -> None:
        """Ctrl+C: copy selected text if any, otherwise quit."""
        try:
            screen = self.screen
            if hasattr(screen, "query_one"):
                log = screen.query_one("#chat-log")
                selection = log.get_selection()
                if selection and selection.strip():
                    import subprocess
                    subprocess.run(["pbcopy"], input=selection.encode(), check=True)
                    # Clear selection after copy
                    log._sel_start = None
                    log._sel_end = None
                    log._selected_text = ""
                    log.refresh()
                    self.notify("Copied!", timeout=1)
                    return
        except Exception:
            pass
        self.action_quit()

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
        # Suppress stdout/stderr FIRST — before cancelling workers
        # so any output from dying worker threads goes to devnull
        devnull = open(os.devnull, "w")
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            # Cancel workers and wait for them to finish
            self.workers.cancel_all()
            for worker in list(self.workers):
                try:
                    worker.wait(timeout=3)
                except Exception:
                    pass
            self._cleanup()
        except Exception:
            pass
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            devnull.close()
        self.exit()


def main() -> None:
    """Entry point for lc-tui or localcode --tui."""
    app = LocalCodeTUI(show_mode_picker=True)
    app.run()


if __name__ == "__main__":
    main()
