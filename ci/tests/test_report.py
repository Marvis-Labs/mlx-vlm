"""The comment renders one shape from pending to done, and never crashes on a
malformed result -- one bad artifact must not take the whole comment down."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ci import report as RP  # noqa: E402

CELLS = [
    {
        "id": "gemma2.apc.off.single",
        "arch": "gemma2",
        "runs_on": ["self-hosted", "macos", "arm64", "mem-16"],
    }
]


def _res(cid, **kw):
    base = {
        "cell": {"id": cid},
        "device": {"chip": "M5", "memory_gb": 128},
        "delta": {},
        "ok": True,
    }
    base.update(kw)
    return base


def test_pending_and_done_have_the_same_frame():
    pending = RP.render("1", CELLS)
    done = RP.render("1", CELLS, [_res("gemma2.apc.off.single")])
    for frame in ("### `gemma2`", "worse ", "per-cell detail"):
        assert frame in pending and frame in done


def test_missing_device_does_not_crash():
    RP.render(
        "1", CELLS, [{"cell": {"id": "gemma2.apc.off.single"}, "delta": {}, "ok": True}]
    )


def test_missing_runs_on_does_not_crash():
    RP.render("1", [{"id": "x", "arch": "gemma2"}])


def test_failed_cell_shows_reason():
    out = RP.render(
        "1",
        CELLS,
        [_res("gemma2.apc.off.single", ok=False, errors=["crashed: expand_dims()"])],
    )
    assert "❌" in out and "crashed: expand_dims()" in out


def test_declined_cell_is_busy_not_failed():
    out = RP.render(
        "1",
        CELLS,
        [
            _res(
                "gemma2.apc.off.single",
                ok=False,
                declined=True,
                errors=["device busy: VirtualMachine is using the machine"],
            )
        ],
    )
    # A decline is environmental: shown apart from a crash, counted as busy,
    # and never as a failure that would read like a broken PR. (The legend
    # still names ❌, so assert on the counts, not on the glyph's absence.)
    assert "⊘" in out
    assert "1 busy" in out and "0 failed" in out
    assert "all devices busy" in out


def test_empty_route_explains_itself():
    out = RP.render("1", [], notes=["gemma2: no sized model"])
    assert "nothing to run" in out and "no sized model" in out


def test_unknown_cell_result_is_ignored():
    RP.render("1", CELLS, [_res("gemma2.GHOST.single")])
