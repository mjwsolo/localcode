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


def test_already_downloaded_short_circuits(isolated_model_dir, monkeypatch):
    choice = catalog.by_key("gemma-12b")
    # Pretend the file is already on disk.
    (isolated_model_dir / choice.filename).write_bytes(b"fake gguf")
    called = {"net": False}
    monkeypatch.setattr(bootstrap, "_try_hf_transfer_download",
                        lambda *a, **k: called.__setitem__("net", True) or True)

    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert choice.filename in path
    assert called["net"] is False, "should not hit the network when file exists"


def test_fast_path_success(isolated_model_dir, monkeypatch):
    choice = catalog.by_key("gemma-12b")

    def _fake_hf(c, model_file, on_progress=None):
        Path(model_file).write_bytes(b"downloaded")
        return True

    monkeypatch.setattr(bootstrap, "_try_hf_transfer_download", _fake_hf)
    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert Path(path).exists()


def test_falls_back_to_urllib_when_fast_path_fails(isolated_model_dir, monkeypatch):
    choice = catalog.by_key("gemma-12b")

    def _hf_boom(*a, **k):
        raise ConnectionError("transient network blip")  # retryable category

    def _fake_parallel(url, dest, **k):
        Path(dest).write_bytes(b"via urllib")

    monkeypatch.setattr(bootstrap, "_try_hf_transfer_download", _hf_boom)
    monkeypatch.setattr(bootstrap, "_download_parallel", _fake_parallel)
    ok, path = bootstrap.download_model(choice)
    assert ok is True
    assert Path(path).read_bytes() == b"via urllib"


def test_fatal_error_fails_fast_without_retry(isolated_model_dir, monkeypatch):
    choice = catalog.by_key("gemma-12b")
    attempts = {"n": 0}

    def _hf_auth_fail(*a, **k):
        attempts["n"] += 1
        raise Exception("401 Unauthorized")

    monkeypatch.setattr(bootstrap, "_try_hf_transfer_download", _hf_auth_fail)
    # If urllib were tried it would also count — assert it's NOT reached.
    monkeypatch.setattr(bootstrap, "_download_parallel",
                        lambda *a, **k: pytest.fail("urllib should not run on auth error"))

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
    choice = min(catalog.CHOICES, key=lambda c: c.size_gb)
    ok, path = bootstrap.download_model(choice)
    assert ok is True, path
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1_000_000
