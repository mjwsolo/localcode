"""Gem live display — real-time status dashboard during model generation.

Shows:
  - Phase indicator with elapsed time
  - Streaming markdown output with syntax highlighting
  - Active tool calls with icons and status
  - Token counter
  - Task checklist for agent mode
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .ui_art import SPINNER_FRAMES, tool_icon


# ── Data model ───────────────────────────────────────────────────────────

@dataclass
class ToolCallStatus:
    name: str
    args_preview: str
    status: str = "running"  # running | done | error
    result_preview: str = ""
    started_at: float = 0.0

    @property
    def elapsed(self) -> str:
        if self.started_at == 0:
            return ""
        return f"{time.time() - self.started_at:.1f}s"


@dataclass
class AgentTask:
    label: str
    status: str = "pending"  # pending | running | done | error


@dataclass
class DisplayState:
    """Mutable state for the live display."""
    phase: str = "idle"              # idle | thinking | tool_call | streaming | done
    phase_detail: str = ""
    started_at: float = 0.0
    content_chunks: list[str] = field(default_factory=list)
    thinking_text: str = ""
    thinking_visible: bool = False
    tool_calls: list[ToolCallStatus] = field(default_factory=list)
    agent_tasks: list[AgentTask] = field(default_factory=list)
    token_count: int = 0
    tick: int = 0
    error: str = ""

    @property
    def elapsed(self) -> str:
        if self.started_at == 0:
            return ""
        secs = time.time() - self.started_at
        if secs < 60:
            return f"{secs:.0f}s"
        mins = int(secs // 60)
        remaining = int(secs % 60)
        return f"{mins}m {remaining:02d}s"

    @property
    def content(self) -> str:
        return "".join(self.content_chunks)

    def reset(self) -> None:
        self.phase = "idle"
        self.phase_detail = ""
        self.started_at = 0.0
        self.content_chunks.clear()
        self.thinking_text = ""
        self.thinking_visible = False
        self.tool_calls.clear()
        self.token_count = 0
        self.tick = 0
        self.error = ""


# ── Renderers ────────────────────────────────────────────────────────────

def _render_phase_header(state: DisplayState) -> Text:
    """The animated phase indicator line."""
    state.tick += 1
    frame_idx = state.tick % len(SPINNER_FRAMES)

    phase_styles = {
        "idle": ("dim", "waiting"),
        "thinking": ("bright_cyan", "reasoning"),
        "tool_call": ("yellow", "executing tools"),
        "streaming": ("bright_cyan", "generating"),
        "verifying": ("yellow", "running tests"),
        "done": ("green", "complete"),
        "error": ("red", "error"),
    }
    style, default_label = phase_styles.get(state.phase, ("dim", state.phase))
    label = state.phase_detail or default_label

    # Animated spinner for active phases
    if state.phase in ("thinking", "tool_call", "streaming", "verifying"):
        spinner_char = SPINNER_FRAMES[frame_idx]
        elapsed = f" ({state.elapsed})" if state.elapsed else ""
        tokens = f" · ↑ {state.token_count:,} tokens" if state.token_count > 0 else ""
        line = Text()
        line.append(f"  {spinner_char} ", style=style)
        line.append(label, style=f"bold {style}")
        line.append(elapsed, style="dim")
        line.append(tokens, style="dim")
        return line

    # Static for done/idle
    icon = "✔" if state.phase == "done" else "✖" if state.phase == "error" else "○"
    line = Text()
    line.append(f"  {icon} ", style=style)
    line.append(label, style=style)
    if state.elapsed:
        line.append(f" ({state.elapsed})", style="dim")
    return line


def _render_tool_calls(state: DisplayState) -> Panel | None:
    if not state.tool_calls:
        return None
    lines = []
    for tc in state.tool_calls[-8:]:  # show last 8
        icon = tool_icon(tc.name)
        status_icon = {
            "running": "[yellow]...[/]",
            "done": "[green]✔[/]",
            "error": "[red]✖[/]",
        }.get(tc.status, "")
        elapsed = f" [dim]{tc.elapsed}[/]" if tc.elapsed else ""
        preview = f"  [dim]{tc.args_preview[:80]}[/]" if tc.args_preview else ""
        result = ""
        if tc.status == "done" and tc.result_preview:
            result = f"\n    [dim]→ {tc.result_preview[:100]}[/]"
        elif tc.status == "error" and tc.result_preview:
            result = f"\n    [red]→ {tc.result_preview[:100]}[/]"
        lines.append(f"  {status_icon} {icon} [bold]{tc.name}[/]{elapsed}{preview}{result}")
    return Panel("\n".join(lines), title="[bold]tools[/]", border_style="yellow", padding=(0, 1))


def _render_agent_tasks(state: DisplayState) -> Panel | None:
    if not state.agent_tasks:
        return None
    lines = []
    for task in state.agent_tasks:
        icon = {
            "pending": "[dim]○[/]",
            "running": "[yellow]◉[/]",
            "done": "[green]✔[/]",
            "error": "[red]✖[/]",
        }.get(task.status, "[dim]○[/]")
        style = {
            "pending": "dim",
            "running": "bold",
            "done": "green",
            "error": "red",
        }.get(task.status, "dim")
        lines.append(f"  {icon} [{style}]{task.label}[/]")
    return Panel("\n".join(lines), title="[bold]tasks[/]", border_style="cyan", padding=(0, 1))


def _render_thinking(state: DisplayState) -> Panel | None:
    if not state.thinking_visible or not state.thinking_text:
        return None
    preview = " ".join(state.thinking_text.split())
    if len(preview) > 300:
        preview = preview[:300] + "..."
    return Panel(
        f"[dim]{preview}[/]",
        title="[dim]thinking[/]",
        border_style="dim",
        padding=(0, 1),
    )


def build_display(state: DisplayState) -> Group:
    """Build the full live display from current state."""
    parts = []

    # Phase header (always shown)
    parts.append(_render_phase_header(state))

    # Agent task list (if in agent mode)
    tasks_panel = _render_agent_tasks(state)
    if tasks_panel:
        parts.append(tasks_panel)

    # Thinking preview
    thinking_panel = _render_thinking(state)
    if thinking_panel:
        parts.append(thinking_panel)

    # Active tool calls
    tools_panel = _render_tool_calls(state)
    if tools_panel:
        parts.append(tools_panel)

    # Streaming content (rendered as markdown)
    content = state.content
    if content and state.phase == "streaming":
        try:
            md = Markdown(content)
            parts.append(Panel(md, border_style="bright_cyan", padding=(0, 1)))
        except Exception:
            parts.append(Panel(content, border_style="bright_cyan", padding=(0, 1)))

    # Error
    if state.error:
        parts.append(Panel(f"[red]{state.error}[/]", title="[red]error[/]", border_style="red"))

    return Group(*parts)


# ── Live display manager ─────────────────────────────────────────────────

class GemLiveDisplay:
    """Manages the Rich Live display for real-time status updates.

    Usage:
        display = GemLiveDisplay(console)
        with display:
            display.start_thinking()
            display.add_content("Hello ")
            display.start_tool("read_file", "path=foo.py")
            display.finish_tool(0, "file contents...")
            display.add_content("world")
            display.finish()
    """

    def __init__(self, console: Console, thinking_mode: str = "summary") -> None:
        self.console = console
        self.thinking_mode = thinking_mode
        self.state = DisplayState()
        self._live: Live | None = None

    def __enter__(self) -> GemLiveDisplay:
        self.state.reset()
        self.state.started_at = time.time()
        self._live = Live(
            build_display(self.state),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def _refresh(self) -> None:
        if self._live:
            self._live.update(build_display(self.state))

    # -- Phase transitions --

    def start_thinking(self, detail: str = "") -> None:
        self.state.phase = "thinking"
        self.state.phase_detail = detail or "reasoning"
        self._refresh()

    def start_streaming(self, detail: str = "") -> None:
        self.state.phase = "streaming"
        self.state.phase_detail = detail or "generating"
        self._refresh()

    def start_verifying(self) -> None:
        self.state.phase = "verifying"
        self.state.phase_detail = "running tests"
        self._refresh()

    def finish(self, detail: str = "") -> None:
        self.state.phase = "done"
        self.state.phase_detail = detail or "complete"
        self._refresh()

    def set_error(self, error: str) -> None:
        self.state.phase = "error"
        self.state.error = error
        self._refresh()

    # -- Content --

    def add_content(self, chunk: str) -> None:
        self.state.content_chunks.append(chunk)
        self.state.token_count += max(1, len(chunk) // 4)  # rough estimate
        if self.state.phase != "streaming":
            self.state.phase = "streaming"
        self._refresh()

    def add_thinking(self, text: str) -> None:
        self.state.thinking_text = text
        if self.thinking_mode != "hidden":
            self.state.thinking_visible = True
        self._refresh()

    # -- Tool calls --

    def start_tool(self, name: str, args_preview: str = "") -> int:
        """Register a tool call starting. Returns index for finish_tool."""
        tc = ToolCallStatus(
            name=name,
            args_preview=args_preview,
            status="running",
            started_at=time.time(),
        )
        self.state.tool_calls.append(tc)
        self.state.phase = "tool_call"
        self.state.phase_detail = f"calling {name}"
        self._refresh()
        return len(self.state.tool_calls) - 1

    def finish_tool(self, index: int, result_preview: str = "", error: bool = False) -> None:
        if 0 <= index < len(self.state.tool_calls):
            tc = self.state.tool_calls[index]
            tc.status = "error" if error else "done"
            tc.result_preview = result_preview
        # If no more running tools, go back to streaming
        if not any(tc.status == "running" for tc in self.state.tool_calls):
            self.state.phase = "streaming"
            self.state.phase_detail = "generating"
        self._refresh()

    # -- Agent tasks --

    def set_agent_tasks(self, labels: list[str]) -> None:
        self.state.agent_tasks = [AgentTask(label=label) for label in labels]
        self._refresh()

    def update_agent_task(self, index: int, status: str) -> None:
        if 0 <= index < len(self.state.agent_tasks):
            self.state.agent_tasks[index].status = status
        self._refresh()
