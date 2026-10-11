"""Model download/install coverage.

Real model downloads are 7–38 GB — impossible to run on every push. So we
split coverage:

  * The DECISION of which model a machine gets is covered by
    test_comprehensive_machines.py (recommend() across the RAM ladder).
  * The download ORCHESTRATION — already-present short-circuit, fast path,
    urllib fallback, retry vs fast-fail on fatal errors, error
    classification — is covered HERE with the network layer mocked, so it
    runs in milliseconds and deterministically.
  * An optional, opt-in REAL download (gated by env var) proves the live
    HuggingFace path actually works end-to-end on demand.
"""
from __future__ import annotations

import os
import sys
import dataclasses
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import bootstrap
from localcode import models_catalog as catalog


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    """Point the catalog's model_dir at a throwaway directory."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(catalog, "model_dir", lambda: d)
    return d


# ── Error classification (pure logic) ───────────────────────────────


@pytest.mark.parametrize("exc,expected", [
    (OSError("No space left on device"), "disk_full"),
    (Exception("401 Unauthorized"), "auth"),
    (Exception("404 Not Found"), "not_found"),
])
def test_error_classification(exc, expected):
    assert bootstrap._classify_download_error(exc) == expected


# ── Orchestration paths (network mocked) ────────────────────────────


def _unpinned(choice):
    """A catalog entry with its integrity pins stripped.

    These tests exercise download ORCHESTRATION with fake byte content, so the
    real sha256/size_bytes pins (added so a stale upstream re-publish is
    caught) would correctly reject every stub. Strip them here; integrity is
    covered by its own tests.
    """
    return dataclasses.replace(choice, sha256=None, size_bytes=None)



def test_already_downloaded_short_circuits(isolated_model_dir, monkeypatch):
    choice = _unpinned(catalog.by_key("gemma-12b"))
    # Pretend the file is already on disk AT FULL SIZE. A tiny stub no
    # longer counts as downloaded — partial files at the final name must
    # fall through to a resume (see
    # test_partial_at_final_name_is_not_treated_as_done).
    (isolated_model_dir / choice.filename).write_bytes(b"fake gguf")
    monkeypatch.setattr(bootstrap, "_is_complete_download", lambda p, e: True)
    called = {"net": False}
    monkeypatch.setattr(bootstrap, "_download_parallel",
                        lambda *a, **k: called.__setitem__("net", True) or True)

    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert choice.filename in path
    assert called["net"] is False, "should not hit the network when file exists"


def test_fast_path_success(isolated_model_dir, monkeypatch):
    choice = _unpinned(catalog.by_key("gemma-12b"))

    def _fake_hf(c, model_file, **kwargs):
        Path(model_file).write_bytes(b"downloaded")
        return True

    monkeypatch.setattr(bootstrap, "_download_parallel", _fake_hf)
    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert Path(path).exists()


def test_parallel_transport_retries_without_deleting_partial(isolated_model_dir, monkeypatch):
    choice = _unpinned(catalog.by_key("gemma-12b"))
    attempts = []
    def transfer(url, dest, **kwargs):
        partial = dest.with_name(dest.name + ".part")
        attempts.append(url)
        if len(attempts) == 1:
            partial.write_bytes(b"checkpoint")
            raise ConnectionError("temporary network interruption")
        assert partial.read_bytes() == b"checkpoint"
        dest.write_bytes(b"complete")
    monkeypatch.setattr(bootstrap, "_download_parallel", transfer)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _: None)
    ok, path = bootstrap.download_model(choice)
    assert ok and Path(path).read_bytes() == b"complete"
    assert len(attempts) == 2


def test_fatal_error_fails_fast_without_retry(isolated_model_dir, monkeypatch):
    choice = _unpinned(catalog.by_key("gemma-12b"))
    attempts = {"n": 0}

    def _hf_auth_fail(*a, **k):
        attempts["n"] += 1
        raise Exception("401 Unauthorized")

    monkeypatch.setattr(bootstrap, "_download_parallel", _hf_auth_fail)

    ok, msg = bootstrap.download_model(choice)
    assert ok is False
    assert attempts["n"] == 1, "auth failure must not retry"


# ── Opt-in REAL download (gated; skipped by default) ────────────────


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("LOCALCODE_RUN_REAL_DOWNLOAD") != "1",
    reason="set LOCALCODE_RUN_REAL_DOWNLOAD=1 to actually download a model (multi-GB)",
)
def test_real_download_smoke(tmp_path, monkeypatch):
    """Actually download the smallest catalog model into a temp dir."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(catalog, "model_dir", lambda: d)
    choice = _unpinned(min(catalog.CHOICES, key=lambda c: c.size_gb))
    ok, path = bootstrap.download_model(choice)
    assert ok is True, path
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1_000_000


def test_partial_at_final_name_is_not_treated_as_done(isolated_model_dir, monkeypatch):
    """Regression (2026-06-11): a partial download at the FINAL filename
    short-circuited download_model as success — llama-server then failed
    on a truncated GGUF. A too-small file must fall through to the
    download, which resumes/replaces it."""
    choice = _unpinned(catalog.by_key("gemma-12b"))
    # A few bytes where a ~7 GB model should be.
    choice.local_path.write_bytes(b"partial junk")
    completed = {"hub": False}

    def _fake_hub(c, model_file, **kwargs):
        # Simulate the hub finishing the download at full size.
        Path(model_file).write_bytes(b"x" * 1024)
        monkeypatch.setattr(bootstrap, "_is_complete_download", lambda p, e: True)
        completed["hub"] = True
        return True

    monkeypatch.setattr(bootstrap, "_download_parallel", _fake_hub)
    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert completed["hub"] is True, "partial file must NOT short-circuit the download"
