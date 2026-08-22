"""web_fetch must not be an SSRF sink.

The model chooses the URL and the tool never prompts, so a fetch of
`http://169.254.169.254/…` or `http://localhost:8080/admin` is one
model turn away — including via a public URL that 302s inward. These
tests pin the host policy. All DNS and HTTP is mocked; nothing here
touches the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.tools import web_fetch  # noqa: E402


class _Ctx:
    pass


class _Resp:
    def __init__(self, status_code=200, headers=None, text="body text"):
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/plain"}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _install_http(monkeypatch, responses):
    """Patch httpx.get with a scripted response map {url: _Resp}.

    Records every URL actually requested so a test can prove the blocked
    hop was never made.
    """
    import httpx

    requested: list[str] = []

    def _get(url, **kwargs):
        requested.append(url)
        assert kwargs.get("follow_redirects") is False, \
            "web_fetch must follow redirects manually so each hop is re-checked"
        if url not in responses:
            raise AssertionError(f"unexpected request to {url}")
        return responses[url]

    monkeypatch.setattr(httpx, "get", _get)
    return requested


def _install_dns(monkeypatch, mapping):
    """Patch socket.getaddrinfo with a {hostname: ip} map."""
    import socket

    def _gai(host, *a, **kw):
        if host not in mapping:
            raise socket.gaierror(f"unknown host {host}")
        ip = mapping[host]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _gai)


# ── literal private / local targets ──────────────────────────────────

BLOCKED_URLS = [
    "http://localhost:8080/admin",
    "http://127.0.0.1/",
    "http://127.0.0.2:3000/x",
    "https://[::1]/",
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://192.168.1.1/",
    "http://10.0.0.5/internal",
    "http://172.16.4.4/",
    "http://100.64.1.1/",          # CGNAT
    "http://[fc00::1]/",           # IPv6 unique-local
    "http://[fe80::1]/",           # IPv6 link-local
    "http://0.0.0.0/",
    "http://printer.local/status",
    "http://api.localhost/",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_private_targets_are_refused(url, monkeypatch):
    requested = _install_http(monkeypatch, {})  # any request is a failure
    out = web_fetch.execute(_Ctx(), {"url": url})
    assert out.startswith("Error:"), out
    assert "local/private host" in out
    assert requested == [], f"a request was made to a blocked host: {requested}"


def test_dns_name_resolving_to_loopback_is_refused(monkeypatch):
    _install_dns(monkeypatch, {"evil.example.com": "127.0.0.1"})
    requested = _install_http(monkeypatch, {})
    out = web_fetch.execute(_Ctx(), {"url": "https://evil.example.com/x"})
    assert out.startswith("Error:")
    assert "local/private host" in out
    assert requested == []


def test_dns_name_resolving_to_metadata_ip_is_refused(monkeypatch):
    _install_dns(monkeypatch, {"meta.example.com": "169.254.169.254"})
    _install_http(monkeypatch, {})
    out = web_fetch.execute(_Ctx(), {"url": "http://meta.example.com/"})
    assert out.startswith("Error:")
    assert "local/private host" in out


def test_non_http_scheme_still_refused(monkeypatch):
    _install_http(monkeypatch, {})
    out = web_fetch.execute(_Ctx(), {"url": "file:///etc/passwd"})
    assert out.startswith("Error: url must start with http(s)://")


# ── redirects ────────────────────────────────────────────────────────


def test_public_url_cannot_redirect_into_localhost(monkeypatch):
    _install_dns(monkeypatch, {"public.example.com": "93.184.216.34"})
    requested = _install_http(monkeypatch, {
        "https://public.example.com/go": _Resp(
            302, {"location": "http://127.0.0.1:8080/admin"}),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://public.example.com/go"})
    assert out.startswith("Error:")
    assert "local/private host" in out
    # The first hop happened; the inward hop must not have.
    assert requested == ["https://public.example.com/go"]


def test_redirect_into_metadata_service_is_blocked(monkeypatch):
    _install_dns(monkeypatch, {"public.example.com": "93.184.216.34"})
    requested = _install_http(monkeypatch, {
        "https://public.example.com/go": _Resp(
            301, {"location": "http://169.254.169.254/latest/meta-data/"}),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://public.example.com/go"})
    assert "local/private host" in out
    assert len(requested) == 1


def test_redirect_to_another_public_host_is_followed(monkeypatch):
    _install_dns(monkeypatch, {
        "a.example.com": "93.184.216.34",
        "b.example.com": "93.184.216.35",
    })
    requested = _install_http(monkeypatch, {
        "https://a.example.com/go": _Resp(302, {"location": "https://b.example.com/final"}),
        "https://b.example.com/final": _Resp(200, {"content-type": "text/plain"}, "arrived"),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://a.example.com/go"})
    assert "arrived" in out
    assert requested == ["https://a.example.com/go", "https://b.example.com/final"]
    # Header reflects the URL actually fetched, not the one requested.
    assert "URL: https://b.example.com/final" in out


def test_relative_redirect_resolves_and_is_checked(monkeypatch):
    _install_dns(monkeypatch, {"a.example.com": "93.184.216.34"})
    requested = _install_http(monkeypatch, {
        "https://a.example.com/go": _Resp(302, {"location": "/landing"}),
        "https://a.example.com/landing": _Resp(200, {"content-type": "text/plain"}, "landed"),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://a.example.com/go"})
    assert "landed" in out
    assert requested[-1] == "https://a.example.com/landing"


def test_redirect_loop_is_capped(monkeypatch):
    _install_dns(monkeypatch, {"loop.example.com": "93.184.216.34"})
    requested = _install_http(monkeypatch, {
        "https://loop.example.com/": _Resp(302, {"location": "https://loop.example.com/"}),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://loop.example.com/"})
    assert "too many redirects" in out
    assert len(requested) == web_fetch._MAX_REDIRECTS + 1


# ── the normal path still works ──────────────────────────────────────


def test_public_host_is_allowed_and_wrapped(monkeypatch):
    _install_dns(monkeypatch, {"docs.example.com": "93.184.216.34"})
    _install_http(monkeypatch, {
        "https://docs.example.com/api": _Resp(
            200, {"content-type": "text/html"},
            "<html><body><h1>Hello</h1><script>bad()</script></body></html>"),
    })
    out = web_fetch.execute(_Ctx(), {"url": "https://docs.example.com/api"})
    assert not out.startswith("Error:")
    assert "URL: https://docs.example.com/api" in out
    assert "Status: 200" in out
    # HTML stripped, script dropped, and the body fenced as untrusted.
    assert "Hello" in out
    assert "bad()" not in out
    assert "UNTRUSTED_DATA" in out


def test_missing_url_argument(monkeypatch):
    out = web_fetch.execute(_Ctx(), {"url": "  "})
    assert out == "Error: 'url' argument is required."


def test_public_ip_literal_is_allowed(monkeypatch):
    _install_http(monkeypatch, {
        "http://93.184.216.34/": _Resp(200, {"content-type": "text/plain"}, "ok body"),
    })
    out = web_fetch.execute(_Ctx(), {"url": "http://93.184.216.34/"})
    assert "ok body" in out
