"""Network status detection — check if internet is available.

Used to gracefully handle offline mode:
- Don't offer web_search when offline
- Let model answer from training knowledge instead
- Cache status for 30 seconds (don't check every turn)
"""
from __future__ import annotations

import socket
import time


_last_check: float = 0
_last_result: bool = True
_cache_ttl: float = 30.0  # seconds


def is_online() -> bool:
    """Check if internet is available. Cached for 30s."""
    global _last_check, _last_result
    now = time.time()
    if now - _last_check < _cache_ttl:
        return _last_result
    _last_check = now
    _last_result = _check_connectivity()
    return _last_result


def _check_connectivity() -> bool:
    """Quick connectivity check — try to resolve DNS."""
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("dns.google", 443)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False
