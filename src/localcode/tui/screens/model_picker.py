"""Model picker screen — HuggingFace-style two-level GGUF browser.

Usable both at setup (before first download) and in-session via /model. The
screen dismisses with the selected `ModelChoice`, or `None` if cancelled.

Two levels, mirroring HuggingFace's GGUF page:

  * Level 1 — a calm list of curated model VERSIONS (``MODEL_GROUPS``), each
    shown as "display_name · maker". Arrow / j-k to move, Enter to open.
  * Level 2 — inside a group, every quant the HF repo ships (fetched live via
    ``hf_quants.fetch_quants``) is rendered as a row: a fit badge for THIS
    Mac's unified memory + the quant label + its exact size, e.g.
    "✓  UD-Q4_K_XL · 7.4 GB". Already-downloaded quants are marked. Enter on a
    quant mints a ``ModelChoice`` via ``choice_for_quant`` and dismisses with
    it — flowing through the EXISTING download/runtime path unchanged.

Layout / styling mirrors the prior picker (and `ModePickerScreen`) so the
visual language stays consistent across pickers.

COMPROMISE NOTE: the delete-downloaded flow (`x` / `y` / `n`) and the
change-save-dir flow (`d`) live at LEVEL 1 only. At level 1 the rows are model
*groups*, not individual downloaded files, so delete here operates over the
curated ``CHOICES`` catalog (the set of files this app knows how to name on
disk) exactly as before — that behavior is preserved verbatim. Delete is NOT
offered at level 2; Esc/left there simply returns to level 1.
"""
from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Input, Static

from ...theme import C


# Fit-badge glyphs (computed via hf_quants.fit_badge → "fits"/"tight"/"too_big").
_BADGE = {
    "fits": "✓",
    "tight": "⚠",
    "too_big": "✗",
}


class ModelPickerScreen(Screen):
    """Two-level model picker.

    Level 1: pick a model version (Enter opens it). Level 2: pick a quant
    (Enter selects → dismiss with a downloadable ModelChoice). Esc cancels at
    level 1, or backs out to level 1 from level 2. Press `d` to change the save
    directory and `x` to delete a downloaded model (both at level 1 only).
    """

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
        Binding("up", "move(-1)", "Previous", show=False),
        Binding("k", "move(-1)", "Previous", show=False),
        Binding("down", "move(1)", "Next", show=False),
        Binding("j", "move(1)", "Next", show=False),
        Binding("enter", "pick_focused", "Open / select focused", show=False),
        Binding("left", "back", "Back", show=False),
        Binding("h", "back", "Back", show=False),
        Binding("d", "edit_dir", "Change save directory", show=False),
        Binding("x", "begin_delete", "Delete a downloaded model", show=False),
        Binding("y", "confirm_delete", "Confirm deletion", show=False),
        Binding("n", "cancel_delete", "Abort deletion", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
    ]

    # Navigation levels:
    #   "groups" — Level 1; list MODEL_GROUPS by version.
    #   "quants" — Level 2; list every quant of the open group.
    _LEVEL_GROUPS = "groups"
    _LEVEL_QUANTS = "quants"

    # Delete flow states (Level 1 only):
    #   "idle"         — default; numbers pick a model group
    #   "awaiting-idx" — user pressed x; next number = which to delete
    #   "awaiting-yn"  — next y/n confirms or aborts the staged delete
    _DELETE_IDLE = "idle"
    _DELETE_AWAIT_IDX = "awaiting-idx"
    _DELETE_AWAIT_YN = "awaiting-yn"

    DEFAULT_CSS = """
    ModelPickerScreen {
        layout: vertical;
        background: ansi_default;
        padding: 1 0;
        background: ansi_default;
    }
    /* #picker-header removed (2026-04-25). Brand moved into footer. */
    #picker-center {
        background: ansi_default;
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    #picker-box {
        background: ansi_default;
        width: 92%;
        max-width: 80;
        height: auto;
        max-height: 90%;        /* cap to viewport so long quant lists don't run off-screen */
        overflow-y: auto;       /* scroll instead of clipping */
        padding: 1 2;
        border: round #5f87ff;
    }
    #picker-list {
        background: ansi_default;
        height: auto;
        width: 100%;
    }
    #dir-input {
        background: ansi_default;
        display: none;
        height: 3;
        margin-top: 1;
        border: round #5f87ff;
    }
    #dir-input.active {
        display: block;
    }
    #picker-footer {
        background: ansi_default;
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

        # ── navigation state ────────────────────────────────────────
        self._level = self._LEVEL_GROUPS
        # Focused index within whichever list is currently shown.
        self._focused_idx = 0
        # The open group (Level 2) and its fetched quants. `None` quants =
        # still loading; `[]` = fetched-but-empty (offline / no quants).
        self._open_group = None
        self._quants = None  # type: list | None

        # ── delete flow state (Level 1) ─────────────────────────────
        self._delete_state = self._DELETE_IDLE
        self._delete_target_idx: int | None = None

        # Live-refresh timer for background-download progress (Level 2). Lazily
        # started when a background download is kicked off; self-cancels once no
        # downloads remain in flight.
        self._progress_timer = None

        # Host memory bandwidth (GB/s), detected once — drives the per-quant
        # tok/s estimate so it reflects THIS machine's chip, not a fixed number.
        try:
            from ...performance import apple_silicon_bandwidth_gbps
            self._bandwidth_gbps = apple_silicon_bandwidth_gbps()
        except Exception:
            self._bandwidth_gbps = 150.0

        # Start Level 1 focus on the user's CURRENTLY-configured model when there
        # is one (so a returning user sees their model highlighted), else on the
        # RAM-recommended group. Map the ModelChoice back to its group via hf_repo.
        from ...models_catalog import MODEL_GROUPS, recommend, current
        try:
            target_repo = None
            try:
                from ...config import load_config
                cur = current(load_config())
                if cur is not None:
                    target_repo = getattr(cur, "hf_repo", None)
            except Exception:
                target_repo = None
            if not target_repo:
                target_repo = recommend().hf_repo
            self._focused_idx = next(
                (i for i, g in enumerate(MODEL_GROUPS) if g.hf_repo == target_repo),
                0,
            )
        except Exception:
            self._focused_idx = 0

    # ── compose ─────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="picker-center"):
            with Vertical(id="picker-box"):
                yield Static(self._render_body(), id="picker-list")
                # can_focus=False while inactive so number keys (1, 2, ...) reach
                # the Screen's BINDINGS dispatcher instead of being captured by
                # this widget. Flipped to True when the user presses `d`.
                dir_input = Input(
                    placeholder="Enter absolute path — Enter saves, Esc cancels",
                    id="dir-input",
                )
                dir_input.can_focus = False
                yield dir_input
        yield Static(self._footer_markup(), id="picker-footer")

    def on_mount(self) -> None:
        return

    def on_resize(self) -> None:
        return

    # ── render dispatch ─────────────────────────────────────────────

    def _render_body(self) -> str:
        if self._level == self._LEVEL_QUANTS:
            return self._render_quants()
        return self._render_groups()

    def _footer_markup(self) -> str:
        if self._level == self._LEVEL_QUANTS:
            # Name the open model family in the footer too — the in-list
            # header (g.display_name) scrolls off the top on long quant lists,
            # so without this the user can't tell which model the quants are for.
            fam = self._open_group.display_name if self._open_group else ""
            fam_bit = f"[dim]·[/] [bold]{fam}[/] " if fam else ""
            return (
                f"[{C.primary}]LocalCode[/] {fam_bit}"
                "[dim]· ↑/↓ + Enter to select a quant · "
                "Esc/← back to models[/]"
            )
        return self._default_footer_markup()

    def _default_footer_markup(self) -> str:
        from ...models_catalog import MODEL_GROUPS
        n = len(MODEL_GROUPS)
        if n == 1:
            keys_hint = "Enter to open"
        elif n == 2:
            keys_hint = "↑/↓ + Enter, or press 1 or 2"
        else:
            keys_hint = f"↑/↓ + Enter, or press 1-{n}"
        return (
            f"[{C.primary}]LocalCode[/] · "
            f"[dim]{keys_hint} · d change save dir · "
            f"x delete model · Esc cancel[/]"
        )

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _tildify(p: Path) -> str:
        try:
            return "~/" + str(p.relative_to(Path.home()))
        except ValueError:
            return str(p)

    # ── Level 1: model groups ───────────────────────────────────────

    def _render_groups(self) -> str:
        """The version list. One name line + one dim status line per group."""
        from ...models_catalog import MODEL_GROUPS, recommend

        lines = [f"[bold]{self._title}[/]", ""]

        # Mark the group the recommender would pick on this machine.
        try:
            rec_repo = recommend().hf_repo
        except Exception:
            rec_repo = None

        cur_repo = self._current_repo()
        any_rec = False

        for i, g in enumerate(MODEL_GROUPS, start=1):
            is_rec = (rec_repo is not None and g.hf_repo == rec_repo)
            if is_rec:
                any_rec = True
            rec_marker = "★ " if is_rec else "  "

            focused = (i - 1 == self._focused_idx)
            chevron = "▸" if focused else " "
            name_style = f"[bold]{g.display_name}[/]"

            # Status line — all dim, middle-dot separators: maker, plus a
            # "current" tag if any quant of this group is the active model.
            status_bits = [g.maker]
            if cur_repo is not None and g.hf_repo == cur_repo:
                status_bits.append("current")
            status_line = "[dim]" + " · ".join(status_bits) + "[/]"

            lines.append(f" [dim]{chevron}[/] [bold]{i}.[/] {rec_marker}{name_style}")
            lines.append(f"       {status_line}")

        lines.append("")
        footer_bits = []
        if any_rec:
            footer_bits.append("★ recommended")
        footer_bits.append("Enter → browse quants")
        footer_bits.append("d → change save dir")
        lines.append("[dim]" + "  ·  ".join(footer_bits) + "[/]")
        return "\n".join(lines)

    def _current_repo(self) -> str | None:
        """hf_repo of the user's currently-selected model, if any."""
        if not self._show_current:
            return None
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return None
        try:
            from ...models_catalog import current as current_choice
            cur = current_choice(cfg)
            return cur.hf_repo if cur is not None else None
        except Exception:
            return None

    # ── Level 2: quants of the open group ───────────────────────────

    def _render_quants(self) -> str:
        from ...models_catalog import _system_ram_gb, model_dir
        from ...hf_quants import fit_badge
        from ... import bootstrap

        g = self._open_group
        lines = [
            f"[bold]{g.display_name}[/] [dim]· {g.maker}[/]",
            "",
        ]

        if self._quants is None:
            lines.append("[dim]loading quants…[/]")
            lines.append("")
            lines.append("[dim]Esc/← back to models[/]")
            return "\n".join(lines)

        rows = self._visible_quants()
        if not rows:
            lines.append("[dim]No quants found.[/]")
            lines.append("[dim]Offline, or this repo ships no GGUF quants yet.[/]")
            lines.append("")
            lines.append("[dim]Esc/← back to models[/]")
            return "\n".join(lines)

        ram = _system_ram_gb()
        mdir = model_dir()
        rec_idx = self._recommended_quant_idx(rows, ram)
        for i, q in enumerate(rows):
            badge = _BADGE.get(fit_badge(q.size_gb, ram), "?")
            focused = (i == self._focused_idx)
            chevron = "▸" if focused else " "
            # ★ marks the quant we recommend for THIS machine (biggest that fits).
            star = "★ " if i == rec_idx else "  "

            # Row: "▸ ★ ✓  UD-Q4_K_XL · 7.4 GB" with an optional state tag.
            # State tag is sourced from the background-download registry
            # (download_status / is_download_complete) so an in-flight quant
            # shows live "downloading 23%" / "queued" alongside "downloaded".
            label = f"[bold]{q.label}[/]"
            tail_bits = [f"{q.size_gb:.1f} GB"]
            # Estimated decode speed for THIS machine's chip. Memory-bandwidth
            # bound, so it scales with the host's bandwidth; the ratio between
            # quants of this model is reliable even if the absolute is rough.
            from ...models_catalog import estimate_decode_tok_s
            spd = estimate_decode_tok_s(q.size_gb, g.display_name, self._bandwidth_gbps)
            if spd:
                tail_bits.append(f"~{spd} tok/s")
            state = self._quant_state(bootstrap, q, mdir)
            if state:
                tail_bits.append(state)
            tail = "[dim] · " + " · ".join(tail_bits) + "[/]"
            lines.append(f" [dim]{chevron}[/] {star}{badge}  {label}{tail}")

        lines.append("")
        lines.append(
            f"[dim]★ best for {ram} GB · ✓ fits · ⚠ tight · ✗ too big · "
            "~tok/s est. for your chip   ·   Esc/← back[/]"
        )
        return "\n".join(lines)

    def _quant_state(self, bootstrap, q, mdir: Path) -> str:
        """The per-quant state tag for a row: "downloaded", "downloading 23%",
        "queued", or "" (none). Sourced from the background-download registry,
        falling back to the on-disk check for files downloaded out-of-band."""
        entry = bootstrap.download_status(q.filename)
        if entry is not None:
            status = entry["status"]
            if status == "done":
                return "downloaded"
            if status == "downloading":
                return f"↓ {entry['progress_pct']}%"
            if status == "queued":
                return "queued"
            if status == "failed":
                return "failed"
        # No registry entry (or a terminal one cleared) — defer to disk truth.
        if (mdir / q.filename).is_file():
            return "downloaded"
        return ""

    # The recommended quant must decode at least this fraction of the FASTEST
    # fitting quant's speed. A relative bar (not an absolute tok/s) because the
    # speed estimate is rough and mis-scales between dense and MoE — but its
    # ORDERING is reliable. This excludes the slow full-precision tier
    # (BF16/Q8) that "biggest that fits" used to recommend on big-RAM Macs,
    # while still picking the highest-quality quant that stays responsive.
    _MIN_SPEED_FRACTION = 0.5

    def _recommended_quant_idx(self, rows, ram: int) -> int | None:
        """Index of the quant to recommend: the highest-quality (largest) quant
        that fits ~55% of RAM AND decodes at >= half the fastest fitting
        quant's estimated speed. Falls back to smallest if nothing fits."""
        if not rows:
            return None
        fitting = [i for i, q in enumerate(rows) if q.size_gb <= 0.55 * ram]
        if not fitting:
            return 0  # rows sorted ascending → smallest
        from ...models_catalog import estimate_decode_tok_s
        name = self._open_group.display_name if self._open_group else ""
        speeds = {
            i: (estimate_decode_tok_s(rows[i].size_gb, name, self._bandwidth_gbps) or 0)
            for i in fitting
        }
        fastest = max(speeds.values()) or 0
        if fastest <= 0:
            return max(fitting, key=lambda i: rows[i].size_gb)
        bar = fastest * self._MIN_SPEED_FRACTION
        responsive = [i for i in fitting if speeds[i] >= bar]
        # Largest (highest quality) quant that stays responsive.
        return max(responsive or fitting, key=lambda i: rows[i].size_gb)

    def _focused_line(self) -> int:
        """Line offset of the focused row in the rendered body (2 header lines;
        quants are 1 line each, groups 2 lines each) — used to scroll it into view."""
        if self._level == self._LEVEL_QUANTS:
            return 2 + self._focused_idx
        return 2 + self._focused_idx * 2

    def _scroll_focus_into_view(self) -> None:
        # Scroll ONLY when the focused row is at/past a viewport edge — just
        # enough to bring it into view. The old code scrolled the focused row
        # to the top on EVERY move, so pressing DOWN through already-visible
        # rows yanked the whole list UP (the "down arrow scrolls up" bug).
        try:
            box = self.query_one("#picker-box")
            line = self._focused_line()
            top = int(getattr(box.scroll_offset, "y", 0) or 0)
            height = int(getattr(box.size, "height", 0) or 0) or 12
            if line <= top:
                box.scroll_to(y=max(0, line - 1), animate=False)
            elif line >= top + height - 1:
                box.scroll_to(y=max(0, line - height + 2), animate=False)
        except Exception:
            pass

    def _visible_quants(self) -> list:
        """Quants shown to the user — mmproj sidecars are filtered out."""
        if not self._quants:
            return []
        return [q for q in self._quants if not q.is_mmproj]

    # ── refresh ─────────────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            self.query_one("#picker-list", Static).update(self._render_body())
        except Exception:
            pass
        # Reset footer to the level's default hint (unless mid-delete-flow,
        # where the transient flash is load-bearing).
        if self._delete_state == self._DELETE_IDLE:
            try:
                self.query_one("#picker-footer", Static).update(self._footer_markup())
            except Exception:
                pass
        self._scroll_focus_into_view()

    def _ensure_progress_timer(self) -> None:
        """Start the live-progress refresh interval if it isn't already running.
        Mirrors the set_interval idiom used elsewhere for periodic refreshes."""
        if self._progress_timer is not None:
            return
        self._progress_timer = self.set_interval(0.5, self._tick_progress)

    def _tick_progress(self) -> None:
        """Refresh in-flight quant rows; self-cancel once nothing's downloading."""
        from ... import bootstrap
        if not bootstrap.list_active_downloads():
            if self._progress_timer is not None:
                self._progress_timer.stop()
                self._progress_timer = None
            # One final repaint so a just-finished quant flips to "downloaded".
            if self._level == self._LEVEL_QUANTS:
                self._refresh()
            return
        if self._level == self._LEVEL_QUANTS:
            self._refresh()

    def _flash_footer(self, markup: str) -> None:
        """Replace the footer with a status message. Restored by the next
        `_refresh` call."""
        try:
            self.query_one("#picker-footer", Static).update(markup)
        except Exception:
            pass

    # ── navigation actions ──────────────────────────────────────────

    def action_move(self, delta: int) -> None:
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return
        n = self._current_count()
        if n == 0:
            return
        self._focused_idx = (self._focused_idx + delta) % n
        self._refresh()

    def _current_count(self) -> int:
        from ...models_catalog import MODEL_GROUPS
        if self._level == self._LEVEL_QUANTS:
            return len(self._visible_quants())
        return len(MODEL_GROUPS)

    def action_pick_focused(self) -> None:
        """Enter — open the focused group (Level 1) or select the focused
        quant (Level 2)."""
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return
        self.action_pick(self._focused_idx)

    def action_pick(self, idx: int) -> None:
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return

        if self._level == self._LEVEL_QUANTS:
            self._pick_quant(idx)
            return

        # ── Level 1 ──────────────────────────────────────────────────
        from ...models_catalog import MODEL_GROUPS
        if not (0 <= idx < len(MODEL_GROUPS)):
            return

        # Number keys do double duty during the delete flow.
        if self._delete_state == self._DELETE_AWAIT_IDX:
            self._stage_delete(idx)
            return

        self._open_group_at(idx)

    def _open_group_at(self, idx: int) -> None:
        """Enter Level 2 for the group at `idx` and kick off the quant fetch."""
        from ...models_catalog import MODEL_GROUPS
        g = MODEL_GROUPS[idx]
        self._open_group = g
        self._level = self._LEVEL_QUANTS
        self._quants = None  # show "loading quants…"
        self._focused_idx = 0
        self._reset_delete_state()
        self._refresh()
        # If downloads are already in flight (e.g. started from another quant or
        # screen), keep the rows live while browsing this group.
        from ... import bootstrap
        if bootstrap.list_active_downloads():
            self._ensure_progress_timer()
        # Fetch on a worker thread so the UI stays responsive (and the cache /
        # network call never blocks the event loop). Mirrors setup.py's
        # run_worker(..., thread=True) + call_from_thread idiom.
        self.run_worker(
            lambda repo=g.hf_repo: self._fetch_quants_worker(repo),
            thread=True,
            exclusive=True,
        )

    def _fetch_quants_worker(self, hf_repo: str) -> None:
        from ...hf_quants import fetch_quants
        try:
            quants = fetch_quants(hf_repo)
        except Exception:
            quants = []
        self.app.call_from_thread(self._on_quants_loaded, hf_repo, quants)

    def _on_quants_loaded(self, hf_repo: str, quants: list) -> None:
        # Ignore stale results if the user already backed out or opened
        # another group while this fetch was in flight.
        if (
            self._level != self._LEVEL_QUANTS
            or self._open_group is None
            or self._open_group.hf_repo != hf_repo
        ):
            return
        self._quants = quants
        # Land focus on the recommended quant for this machine (biggest that fits).
        visible = self._visible_quants()
        if visible:
            from ...models_catalog import _system_ram_gb
            self._focused_idx = self._recommended_quant_idx(visible, _system_ram_gb()) or 0
        else:
            self._focused_idx = 0
        self._refresh()

    def _pick_quant(self, idx: int) -> None:
        """Enter on a quant → use it (if downloaded) or start a background
        download (if not).

        Downloaded → mint a ModelChoice via choice_for_quant and dismiss with
        it (used immediately, existing path). Not downloaded → kick off a
        background download and STAY in the picker, with one exception: if no
        model is usable yet (first-run, nothing on disk), we still dismiss with
        the choice so the caller blocks on the download — otherwise the app has
        nothing to run."""
        from ...models_catalog import choice_for_quant
        from ... import bootstrap
        rows = self._visible_quants()
        if not (0 <= idx < len(rows)):
            return
        q = rows[idx]
        choice = choice_for_quant(self._open_group, q.filename, q.size_gb)

        if bootstrap.is_download_complete(choice):
            self.dismiss(choice)
            return

        # Not on disk. If there's no usable model yet, the caller must block on
        # this download — dismiss with the choice (existing foreground path).
        if not self._has_usable_model():
            self.dismiss(choice)
            return

        # A model is already usable — download in the background and stay put so
        # the user can keep working; the row reflects live progress.
        bootstrap.start_background_download(choice)
        self._ensure_progress_timer()
        self._refresh()
        self._flash_footer(
            f"[dim]Downloading {q.label} in the background — pick it again when "
            "it's ready.[/]"
        )

    def _has_usable_model(self) -> bool:
        """True iff a fully-downloaded model is available to run right now."""
        from ...models_catalog import current as current_choice
        from ... import bootstrap
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return False
        try:
            cur = current_choice(cfg)
        except Exception:
            return False
        return cur is not None and bootstrap.is_download_complete(cur)

    def action_back(self) -> None:
        """Left / h — back out of Level 2 to Level 1 (no-op at Level 1)."""
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return
        if self._level == self._LEVEL_QUANTS:
            self._return_to_groups()

    def _return_to_groups(self) -> None:
        from ...models_catalog import MODEL_GROUPS
        prev_repo = self._open_group.hf_repo if self._open_group else None
        self._level = self._LEVEL_GROUPS
        self._open_group = None
        self._quants = None
        # Restore focus to the group we came from.
        self._focused_idx = next(
            (i for i, g in enumerate(MODEL_GROUPS) if g.hf_repo == prev_repo),
            0,
        )
        self._reset_delete_state()
        self._refresh()

    def action_cancel(self) -> None:
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            inp.remove_class("active")
            inp.value = ""
            inp.can_focus = False
            return
        # Esc aborts any pending delete first.
        if self._delete_state != self._DELETE_IDLE:
            self._reset_delete_state()
            self._refresh()
            return
        # At Level 2, Esc backs out to Level 1 rather than cancelling outright.
        if self._level == self._LEVEL_QUANTS:
            self._return_to_groups()
            return
        self.dismiss(None)

    # ── Delete flow (Level 1 only) ──────────────────────────────────
    #
    # Delete operates over the curated CHOICES catalog — the set of files this
    # app knows how to name on disk. At Level 1 the listed rows are model
    # *groups*; pressing x prompts for a group number, then deletes every
    # downloaded CHOICES file belonging to that group's repo. (Quants browsed
    # at Level 2 are deletable only after they've been promoted into a session;
    # delete is intentionally NOT offered at Level 2 — see module docstring.)

    def _group_choices(self, idx: int) -> list:
        """CHOICES entries whose repo matches MODEL_GROUPS[idx]."""
        from ...models_catalog import CHOICES, MODEL_GROUPS
        repo = MODEL_GROUPS[idx].hf_repo
        return [c for c in CHOICES if c.hf_repo == repo]

    def action_begin_delete(self) -> None:
        """Enter delete mode: the next number key selects which model group to
        delete downloaded files from."""
        inp = self.query_one("#dir-input", Input)
        if inp.has_class("active"):
            return
        if self._level != self._LEVEL_GROUPS:
            return  # delete is Level-1 only
        if self._delete_state != self._DELETE_IDLE:
            return
        from ...models_catalog import MODEL_GROUPS
        any_downloaded = any(
            c.local_path.is_file()
            for i in range(len(MODEL_GROUPS))
            for c in self._group_choices(i)
        )
        if not any_downloaded:
            self._flash_footer("[dim]Nothing to delete — no models are downloaded.[/]")
            return
        self._delete_state = self._DELETE_AWAIT_IDX
        n = len(MODEL_GROUPS)
        which = "1 or 2" if n == 2 else (f"1-{n}" if n > 1 else "1")
        self._flash_footer(
            f"[yellow]Delete which model? Press {which}, or Esc to abort.[/]"
        )

    def _stage_delete(self, idx: int) -> None:
        from ...models_catalog import MODEL_GROUPS
        if not (0 <= idx < len(MODEL_GROUPS)):
            return
        g = MODEL_GROUPS[idx]
        downloaded = [c for c in self._group_choices(idx) if c.local_path.is_file()]
        if not downloaded:
            self._flash_footer(
                f"[yellow]{g.display_name} isn't downloaded — nothing to delete.[/]"
            )
            self._reset_delete_state()
            return
        self._delete_target_idx = idx
        self._delete_state = self._DELETE_AWAIT_YN
        total_gb = sum(c.local_path.stat().st_size for c in downloaded) / (1024 ** 3)
        n = len(downloaded)
        what = downloaded[0].name if n == 1 else f"{n} {g.display_name} files"
        self._flash_footer(
            f"[yellow]Delete {what} ({total_gb:.1f} GB)? "
            "Press y to confirm, n to abort.[/]"
        )

    def action_confirm_delete(self) -> None:
        """Handle `y` — only when awaiting a y/n confirmation on a staged target."""
        if self._delete_state != self._DELETE_AWAIT_YN:
            return
        from ...models_catalog import MODEL_GROUPS, current as current_choice
        from ...config import save_config
        cfg = getattr(self.app, "config", None)
        idx = self._delete_target_idx
        if idx is None or not (0 <= idx < len(MODEL_GROUPS)):
            self._reset_delete_state()
            return

        cur = current_choice(cfg) if cfg is not None else None
        downloaded = [c for c in self._group_choices(idx) if c.local_path.is_file()]
        deleted: list[str] = []
        was_current = False
        try:
            for c in downloaded:
                if c.local_path.is_file():
                    c.local_path.unlink()
                    deleted.append(c.filename)
                    if cur is not None and cur.key == c.key:
                        was_current = True
        except Exception as e:
            self._reset_delete_state()
            self._refresh()
            self._flash_footer(f"[red]Delete failed: {e}[/]")
            return

        if not deleted:
            self._reset_delete_state()
            self._refresh()
            self._flash_footer("[dim]Nothing to delete.[/]")
            return

        msg = f"[green]Deleted {len(deleted)} file(s).[/]"
        if was_current and cfg is not None:
            cfg.runtime.model = ""
            try:
                save_config(cfg)
            except Exception:
                pass
            msg += " [dim]That was the current model — select another to use it.[/]"
        self._reset_delete_state()
        self._refresh()
        self._flash_footer(msg)

    def action_cancel_delete(self) -> None:
        """Handle `n` — abort a pending y/n confirmation."""
        if self._delete_state == self._DELETE_AWAIT_YN:
            self._reset_delete_state()
            self._flash_footer("[dim]Delete cancelled.[/]")

    def _reset_delete_state(self) -> None:
        self._delete_state = self._DELETE_IDLE
        self._delete_target_idx = None

    # ── change-save-dir flow (Level 1 only) ─────────────────────────

    def action_edit_dir(self) -> None:
        from ...models_catalog import model_dir
        if self._level != self._LEVEL_GROUPS:
            return  # save-dir editing is Level-1 only
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
        self._refresh()
