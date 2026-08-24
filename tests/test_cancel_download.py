"""cancel_download: abort an in-flight background download and clean up.

These exercise bootstrap's cancellation directly (no Textual, no network) — a
mocked/registry-marked download entry plus a partial file on disk, then a call
to cancel_download. They assert the cancel Event fires, the registry entry is
dropped, and the partial .gguf is deleted so a later run can never mistake it
for a complete model.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import bootstrap


def _seed_entry(key: str, choice, status: str = "downloading") -> threading.Event:
    """Register a fake in-flight download and return its cancel Event."""
    event = threading.Event()
    with bootstrap._DOWNLOAD_LOCK:
        bootstrap._DOWNLOADS[key] = {
            "model_key": key,
            "name": choice.name,
            "progress_pct": 23,
            "downloaded_mb": 100,
            "total_mb": 7100,
            "status": status,
            "error": None,
        }
        bootstrap._DOWNLOAD_CHOICES[key] = choice
        bootstrap._DOWNLOAD_CANCEL[key] = event
    return event


def _cleanup(key: str) -> None:
    with bootstrap._DOWNLOAD_LOCK:
        bootstrap._DOWNLOADS.pop(key, None)
        bootstrap._DOWNLOAD_CHOICES.pop(key, None)
        bootstrap._DOWNLOAD_CANCEL.pop(key, None)
        if key in bootstrap._DOWNLOAD_QUEUE:
            bootstrap._DOWNLOAD_QUEUE.remove(key)


def _fake_choice(tmp_path: Path, name: str = "fake-model-cancel-Q4_K_M.gguf"):
    return SimpleNamespace(
        filename=name,
        name="Fake Cancel Model",
        local_path=tmp_path / name,
        size_gb=7.1,
        hf_repo="fake/repo",
        revision=None,
    )


def test_cancel_download_running_entry_cleans_up(tmp_path):
    choice = _fake_choice(tmp_path)
    key = choice.filename
    # A partial file (final name) and its .part working file on disk.
    (tmp_path / key).write_bytes(b"\0" * 1024)
    (tmp_path / (key + ".part")).write_bytes(b"\0" * 2048)
    event = _seed_entry(key, choice)
    try:
        result = bootstrap.cancel_download(key)
        assert result is True
        # Cancel Event fired so any worker thread would abort.
        assert event.is_set()
        # Registry entry (and its cancel/choice bookkeeping) is gone.
        assert bootstrap.download_status(key) is None
        with bootstrap._DOWNLOAD_LOCK:
            assert key not in bootstrap._DOWNLOAD_CANCEL
            assert key not in bootstrap._DOWNLOAD_CHOICES
        # Both the partial and the .part working file are deleted.
        assert not (tmp_path / key).exists()
        assert not (tmp_path / (key + ".part")).exists()
    finally:
        _cleanup(key)


def test_cancel_download_when_nothing_downloading_returns_false():
    # No registry entry for this key at all.
    assert bootstrap.cancel_download("no-such-model-xyz.gguf") is False


def test_cancel_download_queued_entry_is_removed_from_queue(tmp_path):
    choice = _fake_choice(tmp_path, "fake-queued-Q4.gguf")
    key = choice.filename
    (tmp_path / key).write_bytes(b"\0" * 512)
    event = _seed_entry(key, choice, status="queued")
    with bootstrap._DOWNLOAD_LOCK:
        bootstrap._DOWNLOAD_QUEUE.append(key)
    try:
        assert bootstrap.cancel_download(key) is True
        assert event.is_set()
        with bootstrap._DOWNLOAD_LOCK:
            assert key not in bootstrap._DOWNLOAD_QUEUE
        assert bootstrap.download_status(key) is None
        assert not (tmp_path / key).exists()
    finally:
        _cleanup(key)


def test_cancel_download_does_not_delete_a_completed_file(tmp_path):
    """A cancel on a just-finished download must NOT delete the good file."""
    choice = _fake_choice(tmp_path, "fake-done-Q4.gguf")
    key = choice.filename
    good = tmp_path / key
    good.write_bytes(b"\0" * 4096)
    _seed_entry(key, choice, status="done")
    try:
        # Nothing in flight (status done) → returns False, file preserved.
        assert bootstrap.cancel_download(key) is False
        assert good.exists()
    finally:
        _cleanup(key)


def test_download_model_aborts_immediately_when_cancel_preset(tmp_path):
    """download_model with an already-set cancel Event returns ("cancelled")
    before touching the network."""
    choice = _fake_choice(tmp_path, "fake-preset-cancel-Q4.gguf")
    choice.size_gb = 0.001  # keep the disk-space preflight trivially satisfied
    event = threading.Event()
    event.set()
    ok, result = bootstrap.download_model(
        choice, on_progress=None, cancel_event=event
    )
    assert ok is False
    assert result == "cancelled"
