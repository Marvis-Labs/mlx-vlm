"""The results store: measured-peak feedback and history."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ci import store  # noqa: E402


def _db():
    return tempfile.mktemp(suffix=".db")


def _rec(db, arch="gemma2", precision="bf16", peak=9.7):
    store.record(
        {
            "cell": {
                "id": f"{arch}.kv_cache.plain.single",
                "arch": arch,
                "component": "kv_cache",
                "precision": precision,
            },
            "device": {"chip": "M3", "memory_gb": 512},
            "ok": True,
            "head": {"peak_mem_gb": {"median": peak}},
            "delta": {},
        },
        "sha",
        db,
    )


def test_measured_peak_round_trips():
    db = _db()
    _rec(db, peak=12.3)
    assert store.measured_peak("gemma2", "bf16", db) == 12.3


def test_measured_peak_absent_is_none():
    assert store.measured_peak("nope", "bf16", _db()) is None


def test_history_returns_entries():
    db = _db()
    _rec(db)
    _rec(db, peak=10.0)
    h = store.history("gemma2.kv_cache.plain.single", db=db)
    assert len(h) == 2 and h[0]["precision"] == "bf16"
