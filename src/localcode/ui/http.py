"""Local service requests never go through an inherited network proxy."""
from __future__ import annotations

import urllib.request
from urllib.parse import urlsplit


def local_urlopen(request, *, timeout=1):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    if urlsplit(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Local service requests require a loopback address")
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def bypass_local_proxies(env: dict[str, str]) -> None:
    entries = []
    for name in ("NO_PROXY", "no_proxy"):
        entries.extend(part.strip() for part in env.get(name, "").split(",") if part.strip())
    value = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost", "::1"]))
    env["NO_PROXY"] = env["no_proxy"] = value
