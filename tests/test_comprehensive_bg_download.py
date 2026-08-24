"""Comprehensive coverage for the BACKGROUND model-download manager.

The provider (bootstrap.py) exposes a small public API that three TUI
screens (setup, model_picker, chat) consume against a shared contract:

    start_background_download(choice) -> model_key: str
    download_status(key) -> dict | None     # the documented shape
    list_active_downloads() -> list[dict]
    is_download_complete(choice) -> bool
    model_key_for(choice) -> str

These tests prove that contract WITHOUT any real network: the actual
byte-mover, ``bootstrap.download_model``, is monkeypatched to a gated fake
so we can assert on the registry/scheduler behaviour deterministically.

The registry lives in module-level globals, so an autouse fixture resets
it around every test to keep them independent.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import bootstrap
from localcode import models_catalog as catalog


# The documented status-dict shape every consumer reads.
_CONTRACT_KEYS = {
    "model_key",
    "name",
    "progress_pct",
    "downloaded_mb",
    "total_mb",
    "status",
    "error",
}


@pytest.fixture(autouse=True)
def reset_registry():
    """Wipe the shared background-download registry around each test."""
    with bootstrap._DOWNLOAD_LOCK:
        bootstrap._DOWNLOADS.clear()
        bootstrap._DOWNLOAD_QUEUE.clear()
        bootstrap._DOWNLOAD_CHOICES.clear()
    yield
    with bootstrap._DOWNLOAD_LOCK:
        bootstrap._DOWNLOADS.clear()
        bootstrap._DOWNLOAD_QUEUE.clear()
        bootstrap._DOWNLOAD_CHOICES.clear()


@pytest.fixture
def not_on_disk(monkeypatch):
    """Force is_download_complete -> False so every start actually enqueues.

    is_download_complete delegates to get_model_path; stubbing get_model_path
    keeps the dependency real (so the is_download_complete-reflects-get_model_path
    test below is meaningful) while guaranteeing a clean 'nothing downloaded'
    baseline for the scheduler tests.
    """
    monkeypatch.setattr(bootstrap, "get_model_path", lambda filename=None: None)


def _choice(key="gemma"):
    return catalog.by_key(key)


def _wait_until(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ── start_background_download: registers + returns a key ─────────────


def test_start_registers_downloading_entry_and_returns_key(monkeypatch, not_on_disk):
    """A started download registers an entry and returns its model_key.

    With the network gated open (download_model blocks), the single start
    occupies a slot and reaches status 'downloading'.
    """
    release = threading.Event()

    def _gated_download(choice, on_progress=None, cancel_event=None):
        release.wait(timeout=3.0)
        return True, str(choice.local_path)

    monkeypatch.setattr(bootstrap, "download_model", _gated_download)

    choice = _choice()
    key = bootstrap.start_background_download(choice)

    assert key == bootstrap.model_key_for(choice) == choice.filename

    # One slot is free, so the entry should be promoted to 'downloading'.
    assert _wait_until(
        lambda: (bootstrap.download_status(key) or {}).get("status") == "downloading"
    )
    release.set()


# ── download_status: documented dict shape ───────────────────────────


def test_download_status_has_documented_shape(monkeypatch, not_on_disk):
    release = threading.Event()
    monkeypatch.setattr(
        bootstrap, "download_model",
        lambda choice, on_progress=None, cancel_event=None: (release.wait(timeout=3.0), (True, str(choice.local_path)))[1],
    )

    choice = _choice()
    key = bootstrap.start_background_download(choice)
    entry = bootstrap.download_status(key)
    release.set()

    assert entry is not None
    assert set(entry.keys()) == _CONTRACT_KEYS
    assert entry["model_key"] == key
    assert entry["name"] == choice.name
    assert isinstance(entry["progress_pct"], int)
    assert isinstance(entry["downloaded_mb"], int)
    assert isinstance(entry["total_mb"], int)
    assert entry["status"] in ("queued", "downloading", "done", "failed")
    # download_status returns a COPY — mutating it must not corrupt the registry.
    entry["status"] = "tampered"
    assert bootstrap.download_status(key)["status"] != "tampered"


def test_download_status_unknown_key_returns_none():
    assert bootstrap.download_status("no-such-model.gguf") is None


# ── concurrency cap: >2 starts leave extras queued ───────────────────


def test_concurrency_cap_leaves_extras_queued(monkeypatch, not_on_disk):
    """More than _MAX_ACTIVE_DOWNLOADS (2) concurrent starts: exactly 2 run,
    the rest sit 'queued' until a slot frees."""
    assert bootstrap._MAX_ACTIVE_DOWNLOADS == 2

    release = threading.Event()

    def _gated_download(choice, on_progress=None, cancel_event=None):
        release.wait(timeout=5.0)
        return True, str(choice.local_path)

    monkeypatch.setattr(bootstrap, "download_model", _gated_download)

    # Four distinct catalog entries -> four distinct registry keys.
    keys = [bootstrap.start_background_download(_choice(k))
            for k in ("gemma", "gemma-12b", "gemma-12b-bf16", "qwen")]
    assert len(set(keys)) == 4

    # Two should reach 'downloading'; the cap holds the other two 'queued'.
    assert _wait_until(
        lambda: sum(
            1 for k in keys
            if (bootstrap.download_status(k) or {}).get("status") == "downloading"
        ) == 2
    )
    statuses = [bootstrap.download_status(k)["status"] for k in keys]
    assert statuses.count("downloading") == 2
    assert statuses.count("queued") == 2

    active = bootstrap.list_active_downloads()
    assert len(active) == 4  # 2 downloading + 2 queued, terminal excluded

    # Free the slots; all four must drain to 'done'.
    release.set()
    assert _wait_until(
        lambda: all(
            (bootstrap.download_status(k) or {}).get("status") == "done" for k in keys
        ),
        timeout=5.0,
    )
    assert bootstrap.list_active_downloads() == []


# ── a completed download flips to done ───────────────────────────────


def test_completed_download_flips_to_done(monkeypatch, not_on_disk):
    """When download_model returns success, the entry transitions to 'done'
    with progress_pct pinned at 100 and no error."""
    monkeypatch.setattr(
        bootstrap, "download_model",
        lambda choice, on_progress=None, cancel_event=None: (True, str(choice.local_path)),
    )

    choice = _choice()
    key = bootstrap.start_background_download(choice)

    assert _wait_until(
        lambda: (bootstrap.download_status(key) or {}).get("status") == "done"
    )
    entry = bootstrap.download_status(key)
    assert entry["status"] == "done"
    assert entry["progress_pct"] == 100
    assert entry["error"] is None


def test_failed_download_records_error(monkeypatch, not_on_disk):
    """A failing download_model leaves the entry 'failed' with the message."""
    monkeypatch.setattr(
        bootstrap, "download_model",
        lambda choice, on_progress=None, cancel_event=None: (False, "boom: network down"),
    )

    choice = _choice()
    key = bootstrap.start_background_download(choice)

    assert _wait_until(
        lambda: (bootstrap.download_status(key) or {}).get("status") == "failed"
    )
    entry = bootstrap.download_status(key)
    assert entry["status"] == "failed"
    assert entry["error"] == "boom: network down"


# ── is_download_complete reflects get_model_path ─────────────────────


def test_is_download_complete_reflects_get_model_path(monkeypatch):
    """is_download_complete must mirror get_model_path: a path -> True,
    None -> False, and it must pass the choice's filename through."""
    choice = _choice()
    seen = {}

    def _present(filename=None):
        seen["filename"] = filename
        return Path("/tmp") / (filename or "x.gguf")

    monkeypatch.setattr(bootstrap, "get_model_path", _present)
    assert bootstrap.is_download_complete(choice) is True
    assert seen["filename"] == choice.filename

    monkeypatch.setattr(bootstrap, "get_model_path", lambda filename=None: None)
    assert bootstrap.is_download_complete(choice) is False


def test_already_complete_registers_done_without_thread(monkeypatch):
    """If the model is already on disk, start registers 'done' immediately and
    never invokes download_model (idempotent short-circuit)."""
    choice = _choice()
    monkeypatch.setattr(
        bootstrap, "get_model_path",
        lambda filename=None: choice.local_path,
    )

    def _must_not_run(*a, **k):  # pragma: no cover - asserts it isn't called
        raise AssertionError("download_model must not run for an on-disk model")

    monkeypatch.setattr(bootstrap, "download_model", _must_not_run)

    key = bootstrap.start_background_download(choice)
    entry = bootstrap.download_status(key)
    assert entry["status"] == "done"
    assert entry["progress_pct"] == 100
    # Terminal 'done' entries are excluded from the active list.
    assert bootstrap.list_active_downloads() == []
