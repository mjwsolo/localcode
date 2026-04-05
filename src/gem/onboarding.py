from __future__ import annotations

from rich.panel import Panel

from .config import AppConfig


def onboarding_panel(config: AppConfig, model_name: str, profile_name: str) -> Panel:
    search_hint = ""
    if config.search.provider == "duckduckgo" and not config.search.brave_api_key:
        search_hint = "\n  tip: better search with brave (free)\n  jem settings set search.provider brave\n  jem settings set search.brave_api_key YOUR_KEY\n  get key: https://brave.com/search/api/"

    body = "\n".join(
        [
            f"profile: {profile_name}",
            f"model:   {model_name}",
            f"mode:    {config.runtime.mode}",
            f"search:  {config.search.provider}",
            "",
            "quick start:",
            "  /help        - commands",
            "  /tools       - available tools",
            "  /agent <task> - agentic mode",
            "  /undo        - revert changes",
            "  /paste       - attach image",
        ]
    ) + search_hint
    return Panel.fit(body, title="start", border_style="green", style="green")
