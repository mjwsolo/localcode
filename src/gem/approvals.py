from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass(slots=True)
class ApprovalItem:
    label: str
    detail: str


class ApprovalQueue:
    def __init__(self, console: Console, prompt: PromptSession) -> None:
        self.console = console
        self.prompt = prompt

    def review(self, title: str, items: list[ApprovalItem], allow_repo_option: bool = False, allow_session_option: bool = True) -> str:
        table = Table("action", "details")
        for item in items:
            table.add_row(item.label, item.detail[:300])
        self.console.print(Panel.fit(table, title=title))
        suffix = "[a]ll once/[n]o/[v]iew"
        if allow_session_option:
            suffix += "/[s]ession-allow"
        if allow_repo_option:
            suffix += "/[r]epo-allow"
        while True:
            raw = self.prompt.prompt(f"Approve {title}? {suffix} ").strip().lower()
            if raw in {"", "a", "all"}:
                return "allow"
            if raw in {"n", "no"}:
                return "deny"
            if raw in {"v", "view"}:
                self.console.print(Panel("\n\n".join(f"{item.label}\n{item.detail}" for item in items), title=f"{title} details"))
                continue
            if allow_session_option and raw in {"s", "session-allow"}:
                return "session-allow"
            if allow_repo_option and raw in {"r", "repo-allow"}:
                return "repo-allow"
