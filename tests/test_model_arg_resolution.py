"""`--model <tag>` must be honored, not silently ignored for the config default.

Regression for the headless bug where the model resolver only accepted a value
ending in `.gguf`. A catalog tag like `qwen38` was skipped, so `localcode
--model qwen38 run` fell through to the configured model (e.g. Qwen 3.6) and
served the WRONG model. `resolve_model_arg` now accepts a tag, a display name,
a bare filename, or a path.
"""

from __future__ import annotations

from pathlib import Path

from localcode import bootstrap
from localcode.models_catalog import CHOICES, by_key


def test_tag_resolves_through_the_catalog(monkeypatch):
    seen = {}

    def fake_get_model_path(fn=None):
        seen["fn"] = fn
        return Path("/models") / fn if fn else None

    monkeypatch.setattr(bootstrap, "get_model_path", fake_get_model_path)
    out = bootstrap.resolve_model_arg("qwen38")
    # The tag was mapped to the catalog entry's real filename, NOT skipped.
    assert seen["fn"] == by_key("qwen38").filename
    assert out == Path("/models") / by_key("qwen38").filename


def test_display_name_resolves(monkeypatch):
    monkeypatch.setattr(
        bootstrap, "get_model_path", lambda fn=None: Path("/m") / fn if fn else None
    )
    choice = CHOICES[0]
    out = bootstrap.resolve_model_arg(choice.name)
    assert out == Path("/m") / choice.filename


def test_bare_gguf_filename_goes_straight_to_disk_lookup(monkeypatch):
    seen = {}

    def fake_get_model_path(fn=None):
        seen["fn"] = fn
        return Path("/models") / fn if fn else None

    monkeypatch.setattr(bootstrap, "get_model_path", fake_get_model_path)
    out = bootstrap.resolve_model_arg("Qwen3.8-27B-UD-Q4_K_XL.gguf")
    assert seen["fn"] == "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    assert out == Path("/models") / "Qwen3.8-27B-UD-Q4_K_XL.gguf"


def test_none_and_unknown_return_none(monkeypatch):
    monkeypatch.setattr(bootstrap, "get_model_path", lambda fn=None: None)
    assert bootstrap.resolve_model_arg(None) is None
    assert bootstrap.resolve_model_arg("") is None
    assert bootstrap.resolve_model_arg("not-a-real-model-xyz") is None
