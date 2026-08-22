"""Download-first path for the three dedicated runners.

The point of this path is that a user with no compiler can still run
diffusion_gemma / cohere2moe / muse_glimmer models. The point of these
tests is that it can never do so by executing an unverified binary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from localcode import bootstrap


def _plat(monkeypatch, system="Darwin", machine="arm64"):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: system)
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: machine)


def _pin(monkeypatch, name, value):
    monkeypatch.setitem(bootstrap._RUNNER_PREBUILT_SHA256, name,
                        {"macos-arm64": value})


def _no_network(monkeypatch):
    """Any actual fetch attempt is a test failure."""
    import urllib.request
    calls = []

    def _boom(*a, **k):
        calls.append(a)
        raise AssertionError("urlretrieve must not be called")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    return calls


def _fake_download(monkeypatch, payload: bytes):
    import urllib.request
    seen = {}

    def _fake(url, filename):
        seen["url"] = url
        Path(filename).write_bytes(payload)
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake)
    return seen


def test_none_pin_never_downloads(tmp_path, monkeypatch):
    _plat(monkeypatch)
    _pin(monkeypatch, "llama-server-muse", None)
    _no_network(monkeypatch)
    ok, reason = bootstrap.download_prebuilt_runner(
        "llama-server-muse", tmp_path / "llama-server-muse")
    assert ok is False
    assert "No pinned checksum" in reason
    assert not (tmp_path / "llama-server-muse").exists()


def test_shipped_pins_are_placeholders_or_hex():
    # A pin must be either None (build from source) or a real sha256 — never
    # a truthy sentinel that would be compared against and always fail, and
    # never something that could be mistaken for "skip verification".
    for name, plats in bootstrap._RUNNER_PREBUILT_SHA256.items():
        for plat, pin in plats.items():
            assert pin is None or (
                isinstance(pin, str) and len(pin) == 64
                and all(c in "0123456789abcdef" for c in pin)
            ), f"{name}/{plat}"


def test_good_download_is_verified_and_executable(tmp_path, monkeypatch):
    _plat(monkeypatch)
    payload = b"#!/bin/sh\necho runner\n"
    digest = hashlib.sha256(payload).hexdigest()
    _pin(monkeypatch, "llama-diffusion-cli", digest)
    seen = _fake_download(monkeypatch, payload)

    dest = tmp_path / "sub" / "llama-diffusion-cli"
    ok, res = bootstrap.download_prebuilt_runner("llama-diffusion-cli", dest)

    assert ok is True and Path(res) == dest
    assert dest.read_bytes() == payload
    assert dest.stat().st_mode & 0o111, "runner must be chmod +x"
    assert seen["url"].startswith("https://github.com/mjwsolo/localcode/releases/download/")
    assert seen["url"].endswith("llama-diffusion-cli-macos-arm64")
    # No temp file left behind.
    assert not dest.with_name(dest.name + ".download").exists()


def test_checksum_mismatch_deletes_and_refuses(tmp_path, monkeypatch):
    _plat(monkeypatch)
    _pin(monkeypatch, "llama-server-cohere", "0" * 64)
    _fake_download(monkeypatch, b"malicious")

    dest = tmp_path / "llama-server-cohere"
    ok, reason = bootstrap.download_prebuilt_runner("llama-server-cohere", dest)

    assert ok is False
    assert "Checksum mismatch" in reason
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".download").exists()


def test_tls_verification_failure_refuses(tmp_path, monkeypatch):
    import ssl
    import urllib.request
    _plat(monkeypatch)
    _pin(monkeypatch, "llama-server-muse", "a" * 64)

    def _bad(url, filename):
        raise ssl.SSLCertVerificationError("unable to get local issuer certificate")

    monkeypatch.setattr(urllib.request, "urlretrieve", _bad)
    dest = tmp_path / "llama-server-muse"
    ok, reason = bootstrap.download_prebuilt_runner("llama-server-muse", dest)
    assert ok is False
    assert "unverified connection" in reason
    assert not dest.exists()


def test_unsupported_platform_does_not_download(tmp_path, monkeypatch):
    _plat(monkeypatch, system="Windows", machine="amd64")
    _pin(monkeypatch, "llama-server-muse", "b" * 64)
    _no_network(monkeypatch)
    ok, reason = bootstrap.download_prebuilt_runner(
        "llama-server-muse", tmp_path / "llama-server-muse")
    assert ok is False and "No prebuilt" in reason


def test_prebuilt_or_note_announces_the_source_build(tmp_path, monkeypatch):
    _plat(monkeypatch)
    _pin(monkeypatch, "llama-server-muse", None)
    msgs: list[str] = []
    ok, res = bootstrap._prebuilt_or_note(
        "llama-server-muse", tmp_path / "llama-server-muse",
        "Muse Glimmer server", msgs.append)
    assert ok is False and res == ""
    joined = " ".join(msgs)
    assert "building from source" in joined
    assert "cmake" in joined


@pytest.mark.parametrize("fn,binname,attr", [
    (lambda: bootstrap.ensure_diffusion_cli, "llama-diffusion-cli", "diffusion"),
    (lambda: bootstrap.ensure_muse_server, "llama-server-muse", "muse"),
])
def test_ensure_tries_download_before_any_toolchain(tmp_path, monkeypatch, fn, binname, attr):
    monkeypatch.setenv("HOME", str(tmp_path))
    _plat(monkeypatch)
    payload = b"#!/bin/sh\nexit 0\n"
    _pin(monkeypatch, binname, hashlib.sha256(payload).hexdigest())
    _fake_download(monkeypatch, payload)
    # No git, no cmake: the download path must not care.
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _n: None)
    monkeypatch.setattr(bootstrap, "_ensure_cmake", lambda: False)

    ok, res = fn()()
    assert ok is True
    dest = tmp_path / ".local" / "share" / "localcode" / binname
    assert Path(res) == dest
    assert dest.stat().st_mode & 0o111


@pytest.mark.parametrize("fn,label", [
    (lambda: bootstrap.ensure_diffusion_cli, "diffusion runner"),
    (lambda: bootstrap.ensure_muse_server, "Muse Glimmer server"),
])
def test_missing_toolchain_names_what_to_install(tmp_path, monkeypatch, fn, label):
    monkeypatch.setenv("HOME", str(tmp_path))
    _plat(monkeypatch)
    _pin(monkeypatch, "llama-diffusion-cli", None)
    _pin(monkeypatch, "llama-server-muse", None)
    _no_network(monkeypatch)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _n: None)

    ok, err = fn()()
    assert ok is False
    assert "xcode-select --install" in err
