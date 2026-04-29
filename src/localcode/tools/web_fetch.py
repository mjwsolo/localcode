"""web_fetch — GET one URL, strip HTML to text, truncate."""
from __future__ import annotations

import re

from .base import ToolContext

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch the content of a specific URL. Unlike web_search (which "
            "returns a list of links), this retrieves and returns the body "
            "of one page — typically an API doc, GitHub README, blog post, "
            "or error-message forum thread. HTML is stripped to readable "
            "text. Truncated at 10 K chars."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL, e.g. https://example.com/docs/api"},
            },
            "required": ["url"],
        },
    },
}


_LIMIT = 10000


def execute(ctx: ToolContext, args: dict) -> str:
    import httpx
    url = (args.get("url") or "").strip()
    if not url:
        return "Error: 'url' argument is required."
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Error: url must start with http(s)://; got {url!r}"
    try:
        r = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "localcode/1.0 (coding-agent)"},
        )
        r.raise_for_status()
    except Exception as e:
        return f"Fetch error: {e}"

    content_type = r.headers.get("content-type", "").lower()
    body = r.text
    if "html" in content_type:
        body = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ",
                      body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"&nbsp;", " ", body)
        body = re.sub(r"\s+", " ", body).strip()

    truncated_note = ""
    if len(body) > _LIMIT:
        truncated_note = (
            f"\n\n[... {len(body) - _LIMIT} more chars truncated; "
            f"fetch a more specific URL or search for the section you want ...]"
        )
        body = body[:_LIMIT]

    # Prompt-injection defence: wrap fetched page content in explicit
    # data/instruction separator markers. The web is a high-risk input
    # source — a malicious page could try "IGNORE ALL PRIOR INSTRUCTIONS"
    # directly, or hide it in a fake code block. The wrapper + signature
    # detector in injection_defense.py is our minimum baseline.
    from ..injection_defense import wrap_untrusted
    header = (
        f"URL: {url}\nStatus: {r.status_code}\n"
        f"Content-Type: {content_type or '(unknown)'}"
    )
    return f"{header}\n\n{wrap_untrusted(body + truncated_note, source=url)}"


def is_concurrency_safe(args: dict) -> bool:
    return True
