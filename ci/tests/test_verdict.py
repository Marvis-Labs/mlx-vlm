"""The judgement: what counts as a regression, and what is noise."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ci import verdict as V  # noqa: E402


def d(change, noise, sig=None, **kw):
    return {
        "change_pct": change,
        "noise_pct": noise,
        "significant": (
            change is not None and abs(change) > noise if sig is None else sig
        ),
        **kw,
    }


def test_change_within_noise_is_noise():
    assert V.verdict("decode_tps", d(-1.0, 3.0)) == "noise"


def test_real_regression_flags():
    assert V.verdict("decode_tps", d(-8.0, 2.0)) == "regressed"


def test_improvement_flags():
    assert V.verdict("decode_tps", d(8.0, 2.0)) == "improved"


def test_floor_blocks_tiny_change_on_zero_noise():
    # peak memory is perfectly repeatable: a 0.4% change must not flag
    assert V.verdict("peak_mem_gb", d(0.4, 0.0)) == "noise"


def test_wide_noise_is_inconclusive():
    assert V.verdict("decode_tps", d(-5.0, 99.0)) == "inconclusive"


def test_nan_is_noise_not_regression():
    assert V.verdict("decode_tps", d(float("nan"), 2.0, sig=True)) == "noise"


def test_zero_baseline_counter_switching_on_is_regression_only_if_functional():
    on = {"change_pct": None, "significant": True, "functional": True}
    off = {"change_pct": None, "significant": True, "functional": False}
    assert V.verdict("token_hit_rate", on) == "regressed"
    assert V.verdict("decode_tps", off) == "noise"


def test_bar_uses_floor_when_noise_is_small():
    assert V.bar_for("decode_tps", 0.1) == V.floor_for("decode_tps")
