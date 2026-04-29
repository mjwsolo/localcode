"""Model picker screen — number-key selection of a GGUF from the catalog.

Usable both at setup (before first download) and in-session via /model. The
screen dismisses with the selected `ModelChoice`, or `None` if cancelled.
Layout mirrors `ModePickerScreen` so the visual language stays consistent
across pickers.
"""
from __future__ import annotations

from ...theme import C


from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Input, Static


class ModelPickerScreen(Screen):
    """Pick a model by number key. Esc cancels. Press `d` to change save directory."""

    BINDINGS = [
        Binding("1", "pick(0)", "#1", show=False),
        Binding("2", "pick(1)", "#2", show=False),
        Binding("3", "pick(2)", "#3", show=False),
        Binding("4", "pick(3)", "#4", show=False),
        Binding("5", "pick(4)", "#5", show=False),
        Binding("6", "pick(5)", "#6", show=False),
        Binding("7", "pick(6)", "#7", show=False),
        Binding("8", "pick(7)", "#8", show=False),
        Binding("9", "pick(8)", "#9", show=False),
        Binding("d", "edit_dir", "Change save directory", show=False),
        Binding("x", "begin_delete", "Delete a downloaded model", show=False),
        Binding("y", "confirm_delete", "Confirm deletion", show=False),
        Binding("n", "cancel_delete", "Abort deletion", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    # Delete flow states:
    #   "idle"         — default; numbers pick a model
    #   "awaiting-idx" — user pressed x; next number = which to delete
    #   "awaiting-yn"  — next y/n confirms or aborts the staged delete
    _DELETE_IDLE = "idle"
    _DELETE_AWAIT_IDX = "awaiting-idx"
    _DELETE_AWAIT_YN = "awaiting-yn"

    DEFAULT_CSS = """
    ModelPickerScreen {
        layout: vertical;
        background: $surface;
        padding: 1 0;
        background: $surface;
    }
    /* #picker-header removed (2026-04-25). Brand moved into footer. */
    #picker-center {
        background: $surface;
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    #picker-box {
        background: $surface;
        width: 92%;
        max-width: 56;
        height: auto;
        padding: 1 2;
        border: round #5f87ff;
    }
    #picker-list {
        background: $surface;
        height: auto;
        width: 100%;
    }
    #dir-input {
        background: $surface;
        display: none;
        height: 3;
        margin-top: 1;
        border: round #5f87ff;
    }
    #dir-input.active {
        display: block;
    }
    #picker-footer {
        background: $surface;
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    def __init__(self, title: str = "Choose a model", show_current: bool = True) -> None:
        super().__init__()
        self._title = title
        self._show_current = show_current
        self._delete_state = self._DELETE_IDLE
        self._delete_target_idx: int | None = None

    def compose(self) -> ComposeResult:
        from ...models_catalog import current as current_choice

        cfg = getattr(self.app, "config", None)
        cur = current_choice(cfg) if (self._show_current and cfg is not None) else None

        # Brand merges into #picker-footer at the bottom — see
        # _default_footer_markup. No separate brand widget on this
        # screen because the footer already occupies the bottom row.
        with Container(id="picker-center"):
            with Vertical(id="picker-box"):
                yield Static(self._render_list(cur), id="picker-list")
                # can_focus=False while inactive so number keys (1, 2, ...) reach
                # the Screen's BINDINGS dispatcher instead of being captured by
                # this widget (even when display: none it was still in the focus
                # chain). We flip it to True when the user presses `d` to edit
                # the save directory.
                dir_input = Input(
                    placeholder="Enter absolute path — Enter saves, Esc cancels",
                    id="dir-input",
                )
                dir_input.can_focus = False
                yield dir_input
        # Footer content comes from _default_footer_markup() — it scales to
        # the actual choice count and lists the available shortcuts.
        yield Static(self._default_footer_markup(), id="picker-footer")

    def _default_footer_markup(self) -> str:
        from ...models_catalog import CHOICES
        from ...theme import C
        n = len(CHOICES)
        if n == 1:
            keys_hint = "press 1 to select"
        elif n == 2:
            keys_hint = "press 1 or 2 to select"
        else:
            keys_hint = f"press 1-{n} to select"
        return (
            f"🏠[{C.primary}]LocalCode[/]  │  "
            f"[dim]{keys_hint}   ·   d change save dir   ·   "
            f"x delete model   ·   Esc cancel[/]"
        )

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _tildify(p: Path) -> str:
        try:
            return "~/" + str(p.relative_to(Path.home()))
        except ValueError:
            return str(p)

    def _render_list(self, cur) -> str:
        from ...models_catalog import CHOICES, model_dir
        lines = [f"[bold]{self._title}[/]", ""]
        # Render each model as TWO intentional lines:
        #   "  1. <name>   <size> GB"
        #   "     ✓ downloaded   (current)"
        # The status indent (5 spaces) lines up under the name (which
        # starts after "  1. " = 5 cols). Previously the status was
        # appended to the name line and wrapped to col 0 when the box
        # was too narrow, producing the left-flush ✓ in the screenshot.
        # Two intentional lines fix the alignment and survive any width.
        for i, c in enumerate(CHOICES, start=1):
            downloaded = c.local_path.is_file()
            status = f"[{C.success}]✓ downloaded[/]" if downloaded else "[yellow]↓ needs download[/]"
            marker = f"   [{C.success}](current)[/]" if cur and cur.key == c.key else ""
            lines.append(
                f"  [bold]{i}.[/] [bold]{c.name}[/]   [dim]{c.size_gb:.1f} GB[/]"
            )
            lines.append(f"     {status}{marker}")
        lines.append("")
        # 'Saves to ...' on one line, hint on its own line so the
        # 'd to change' phrase doesn't wrap into a half-broken sentence.
        lines.append(f"[dim]Saves to {self._tildify(model_dir())}/[/]")
        lines.append("[dim italic](press d to change)[/]")
        return "\n".join(lines)

    def _refresh_list(self) -> None:
        from ...models_catalog import current as current_choice
        cfg = getattr(self.app, "config", None)
        cur = current_choice(cfg) if (self._show_current and cfg is not None) else None
        self.query_one("#picker-list", Static).update(self._render_list(cur))
        # Reset footer to the default hint line after any transient flash
        # (unless we're mid-delete-flow and the flash is load-bearing).
        if self._delete_state == self._DELETE_IDLE:
            try:
                self.query_one("#picker-footer", Static).update(self._default_footer_markup())
            except Exception:
                pass

    # Header bar removed (2026-04-25). Brand now sits at the left of
    # #picker-footer alongside the keyboard hints — see
    # _default_footer_markup. on_mount / on_resize are now no-ops.

    def on_mount(self) -> None:
        return

    def on_resize(self) -> None:
        return

    # ── actions ─────────────────────────────────────────────────────

    def action_pick(self, idx: int) -> None:
        # If the directory input is focused, number keys should type into it.
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return
        from ...models_catalog import CHOICES
        if not (0 <= idx < len(CHOICES)):
            return

        # Number keys do double duty during the delete flow — they pick which
        # model to delete instead of selecting it.
        if self._delete_state == self._DELETE_AWAIT_IDX:
            choice = CHOICES[idx]
            if not choice.local_path.is_file():
                self._flash_footer(
                    f"[yellow]{choice.name} isn't downloaded — nothing to delete.[/]"
                )
                self._reset_delete_state()
                return
            self._delete_target_idx = idx
            self._delete_state = self._DELETE_AWAIT_YN
            size_gb = choice.local_path.stat().st_size / (1024 ** 3)
            self._flash_footer(
                f"[yellow]Delete {choice.name} ({size_gb:.1f} GB)? "
                "Press y to confirm, n to abort.[/]"
            )
            return

        self.dismiss(CHOICES[idx])

    def action_cancel(self) -> None:
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            inp.remove_class("active")
            inp.value = ""
            inp.can_focus = False
            return
        # Esc also aborts any pending delete
        if self._delete_state != self._DELETE_IDLE:
            self._reset_delete_state()
            self._refresh_list()
            return
        self.dismiss(None)

    # ── Delete flow ─────────────────────────────────────────────────

    def action_begin_delete(self) -> None:
        """Enter delete mode: the next number key selects which model to delete."""
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return  # typing a path, ignore
        if self._delete_state != self._DELETE_IDLE:
            return
        from ...models_catalog import CHOICES
        if not any(c.local_path.is_file() for c in CHOICES):
            self._flash_footer("[dim]Nothing to delete — no models are downloaded.[/]")
            return
        self._delete_state = self._DELETE_AWAIT_IDX
        n = len(CHOICES)
        which = "1 or 2" if n == 2 else (f"1-{n}" if n > 1 else "1")
        self._flash_footer(
            f"[yellow]Delete which model? Press {which}, or Esc to abort.[/]"
        )

    def action_confirm_delete(self) -> None:
        """Handle `y` — only when awaiting a y/n confirmation on a staged target."""
        if self._delete_state != self._DELETE_AWAIT_YN:
            return
        from ...models_catalog import CHOICES, current as current_choice
        from ...config import save_config
        cfg = getattr(self.app, "config", None)
        idx = self._delete_target_idx
        if idx is None or not (0 <= idx < len(CHOICES)):
            self._reset_delete_state()
            return
        choice = CHOICES[idx]
        path = choice.local_path
        # Was this the user's current model? Clear the config pointer so the
        # next session doesn't try to launch a file that no longer exists.
        was_current = (
            cfg is not None
            and current_choice(cfg) is not None
            and current_choice(cfg).key == choice.key
        )
        try:
            if path.is_file():
                path.unlink()
        except Exception as e:
            self._flash_footer(f"[red]Delete failed: {e}[/]")
            self._reset_delete_state()
            return
        msg = f"[green]Deleted {choice.filename}.[/]"
        if was_current and cfg is not None:
            cfg.runtime.model = ""
            try:
                save_config(cfg)
            except Exception:
                pass
            msg += " [dim]That was the current model — select another to use it.[/]"
        self._reset_delete_state()
        self._refresh_list()  # entry flips to '↓ needs download'
        self._flash_footer(msg)

    def action_cancel_delete(self) -> None:
        """Handle `n` — abort a pending y/n confirmation."""
        if self._delete_state == self._DELETE_AWAIT_YN:
            self._reset_delete_state()
            self._flash_footer("[dim]Delete cancelled.[/]")

    def _reset_delete_state(self) -> None:
        self._delete_state = self._DELETE_IDLE
        self._delete_target_idx = None

    def _flash_footer(self, markup: str) -> None:
        """Replace the footer with a status message. Restored by the next
        `_refresh_list` call (or on-mount)."""
        try:
            self.query_one("#picker-footer", Static).update(markup)
        except Exception:
            pass

    def action_edit_dir(self) -> None:
        from ...models_catalog import model_dir
        inp = self.query_one("#dir-input", Input)
        inp.value = str(model_dir())
        inp.add_class("active")
        # Only grab focus while actually being edited — see compose() for why.
        inp.can_focus = True
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "dir-input":
            return
        from ...config import save_config
        new_raw = (event.value or "").strip()
        if not new_raw:
            event.input.remove_class("active")
            return
        try:
            new_path = Path(new_raw).expanduser().resolve()
            new_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.notify(f"Couldn't use that path: {e}", severity="error")
            return
        cfg = self.app.config
        cfg.runtime.model_dir = str(new_path)
        try:
            save_config(cfg)
        except Exception as e:
            self.notify(
                f"Saved in-memory but couldn't persist config: {e}",
                severity="warning",
            )
        event.input.remove_class("active")
        event.input.value = ""
        event.input.can_focus = False  # hand focus back to the screen's key bindings
        self._refresh_list()
