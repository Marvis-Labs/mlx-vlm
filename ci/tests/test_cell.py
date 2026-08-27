"""Comparison and on-device variant selection."""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ci import cell as C  # noqa: E402


def _summary(median, se=0.5):
    return {"median": median, "cv_pct": 1.0, "stderr_pct": se}


def test_compare_flags_a_real_change():
    d = C.compare({"decode_tps": _summary(50)}, {"decode_tps": _summary(40)})
    assert d["decode_tps"]["change_pct"] < 0 and d["decode_tps"]["significant"]


def test_compare_tolerates_missing_spread_fields():
    # a truncated or older-format summary lacks stderr_pct -> must not crash
    C.compare({"x": {"median": 50}}, {"x": {"median": 50}})


def test_compare_zero_baseline_counter():
    d = C.compare({"exact_hits": _summary(0)}, {"exact_hits": _summary(5)})
    assert d["exact_hits"]["change_pct"] is None


def test_variant_choice_picks_largest_that_fits():
    cell = {
        "id": "m",
        "variants": [
            {"repo": "big", "revision": "", "precision": "bf16", "requires_gb": 88},
            {"repo": "small", "revision": "", "precision": "4bit", "requires_gb": 20},
        ],
    }
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(stdout=str(128 * 2**30))
        assert C.choose_variant(dict(cell))["precision"] == "bf16"
        m.return_value = mock.Mock(stdout=str(32 * 2**30))
        assert C.choose_variant(dict(cell))["precision"] == "4bit"


def test_variant_choice_declines_when_too_small():
    cell = {
        "id": "m",
        "variants": [
            {"repo": "x", "revision": "", "precision": "4bit", "requires_gb": 20}
        ],
    }
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(stdout=str(16 * 2**30))
        try:
            C.choose_variant(dict(cell))
            assert False, "should have declined"
        except SystemExit:
            pass


def test_probe_error_extracts_real_reason():
    proc = mock.Mock(
        returncode=1, stderr="  File x\nTypeError: expand_dims(): bad args"
    )
    assert C._probe_error(proc) == "crashed: expand_dims(): bad args"
