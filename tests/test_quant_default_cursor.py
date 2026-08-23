"""The quant picker must land the cursor on a downloaded quant when one exists.

The QA pass caught the picker defaulting the cursor to the recommended quant
even when it was NOT on disk, so a naive Enter kicked off a fresh multi-GB
download instead of using an already-local quant. `_default_quant_idx` encodes
the fix: prefer a downloaded row, falling back to the recommendation only when
nothing is downloaded.
"""

from __future__ import annotations

from localcode.tui.screens.model_picker import _default_quant_idx


def test_recommended_is_used_when_it_is_downloaded():
    # rec index 1 is among the downloaded set -> use it directly.
    assert _default_quant_idx([7.4, 10.7, 23.8], rec=1, downloaded=[0, 1]) == 1


def test_falls_back_to_largest_downloaded_at_or_below_recommendation():
    # rec index 1 (8.6 GB) is NOT downloaded; downloaded are 0 (7.4) and 2
    # (23.8). Prefer the largest downloaded that is <= the recommended size,
    # i.e. index 0, NOT the 23.8 GB one.
    assert _default_quant_idx([7.4, 8.6, 23.8], rec=1, downloaded=[0, 2]) == 0


def test_uses_largest_downloaded_when_all_are_above_recommendation():
    # rec 0 (7.4) not downloaded; only bigger quants are local -> largest local.
    assert _default_quant_idx([7.4, 10.7, 23.8], rec=0, downloaded=[1, 2]) == 2


def test_no_downloads_falls_back_to_recommendation():
    assert _default_quant_idx([7.4, 8.6], rec=1, downloaded=[]) == 1


def test_no_downloads_and_no_recommendation_defaults_to_first():
    assert _default_quant_idx([7.4, 8.6], rec=None, downloaded=[]) == 0


def test_empty_rows():
    assert _default_quant_idx([], rec=None, downloaded=[]) == 0
