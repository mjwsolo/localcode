"""Opt-in local inference using the shipping launcher and authenticated API.

For update promotion use scripts/verify_models.py --strict; that fails if a
required model is absent, covers turbo4 and the bundled OpenCode runtime, and
writes a receipt. This pytest tier only exercises downloaded models.
"""
from pathlib import Path

import pytest

from scripts.verify_models import Server, catalog, verify

pytestmark = pytest.mark.real_models
CHOICES = [c for c in catalog.CHOICES if c.local_path.is_file()]


@pytest.mark.parametrize("choice", CHOICES or [None], ids=lambda c: c.key if c else "no-models")
def test_shipping_model_contract(choice, tmp_path: Path):
    if choice is None:
        pytest.skip("no models downloaded; promotion uses the strict CLI gate")
    with Server(choice, tmp_path) as server:
        verify(server, runtime=choice.key in {"qwen38", "openjev"})
