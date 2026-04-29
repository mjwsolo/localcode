"""web_search — DuckDuckGo-backed web search returning top result snippets."""
from __future__ import annotations

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for documentation, APIs, error solutions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


def execute(ctx: ToolContext, args: dict) -> str:
    query = args["query"]
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found"
        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


def is_concurrency_safe(args: dict) -> bool:
    return True
