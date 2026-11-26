"""The starred quant leaves max(8 GB, 20% of RAM) free; the largest fit stays selectable."""
import pytest

from localcode.fit import comfortable, free_gb
from localcode.hf_quants import Quant
from localcode.ui.supervisor import Supervisor

QWEN = "Qwen3.6-35B-A3B-"
QWEN_SIZES = {"UD-IQ2_M": 11.5, "UD-IQ3_S": 13.7, "UD-Q3_K_XL": 16.8, "UD-IQ4_XS": 17.7,
              "UD-Q4_K_S": 20.9, "UD-Q4_K_M": 22.1, "UD-Q4_K_XL": 22.4, "UD-Q5_K_XL": 26.6,
              "UD-Q6_K": 29.3, "UD-Q6_K_XL": 31.8, "Q8_0": 36.9}
GEMMA = "gemma-4-26B-A4B-it-"
GEMMA_SIZES = {"UD-IQ2_M": 10.0, "UD-IQ3_S": 11.3, "UD-Q3_K_XL": 12.9, "UD-Q4_K_M": 16.9,
               "UD-Q4_K_XL": 17.0, "UD-Q5_K_XL": 21.2, "UD-Q6_K": 23.2, "UD-Q8_K_XL": 27.6}


def rows(prefix, sizes):
    return sorted((Quant(f"{prefix}{k}.gguf", v, k, False) for k, v in sizes.items()), key=lambda q: q.size_gb)


def sup(ram):
    s = Supervisor.__new__(Supervisor)
    s.ram_gb = ram
    s.bandwidth = 400.0
    return s


@pytest.mark.parametrize("ram", [16, 24, 32, 36, 48, 64, 128])
@pytest.mark.parametrize("prefix,sizes,name", [(QWEN, QWEN_SIZES, "Qwen3.6 35B A3B"), (GEMMA, GEMMA_SIZES, "Gemma 4 26B A4B")])
def test_star_leaves_headroom_or_is_flagged_tight(ram, prefix, sizes, name):
    s, r = sup(ram), rows(prefix, sizes)
    star, mx = s._recommended_quant_idx(r, name), s._largest_fit_idx(r)
    if star is None:
        assert mx is None
        return
    q = r[star]
    assert q.size_gb <= r[mx].size_gb
    if comfortable(q.size_gb, ram, q.filename):
        assert free_gb(q.size_gb, ram, q.filename) >= max(8, 0.2 * ram)
    else:
        # no quant is comfortable: the star falls back to the largest that fits
        assert star == mx
        assert not any(comfortable(x.size_gb, ram, x.filename) for x in r)


def test_36gb_qwen_star_is_below_q4_k_xl():
    s, r = sup(36), rows(QWEN, QWEN_SIZES)
    star = r[s._recommended_quant_idx(r, "Qwen3.6 35B A3B")]
    assert star.size_gb < QWEN_SIZES["UD-Q4_K_XL"]
    assert r[s._largest_fit_idx(r)].size_gb >= star.size_gb


def test_128gb_star_leaves_a_fifth_free():
    s, r = sup(128), rows(QWEN, QWEN_SIZES)
    star = r[s._recommended_quant_idx(r, "Qwen3.6 35B A3B")]
    assert free_gb(star.size_gb, 128, star.filename) >= 25.6
