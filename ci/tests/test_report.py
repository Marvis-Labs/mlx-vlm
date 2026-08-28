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


def test_parity_applies_committed_thresholds_not_just_greedy():
    # High greedy agreement but KL past the ceiling must fail. The old renderer
    # hardcoded greedy>=0.98 and never looked at KL, so this would have passed
    # silently -- a model that picks the same argmax while diverging in
    # probability. The committed parity_thresholds.yaml must actually apply.
    cells = [
        {
            "id": "smollm3.parity",
            "arch": "smollm3",
            "runs_on": ["self-hosted", "macos", "arm64", "parity"],
        }
    ]
    res = [
        {
            "cell": {
                "id": "smollm3.parity",
                "arch": "smollm3",
                "greedy_agreement": 0.995,
                "kl_mean": 0.005,
                "kl_max": 0.5,
            },
            "device": {"chip": "M4", "memory_gb": 16},
            "delta": {},
            "ok": True,
        }
    ]
    out = RP.render("9", cells, res)
    assert "🔴" in out
    assert "kl_max 0.5 🔴" in out
    # greedy is fine, so it must not be flagged
    assert "greedy 0.995 🔴" not in out


def test_refusal_replaces_the_report():
    out = RP.render(
        "1", [], notes=["REFUSED: this pull request changes protected CI files (x)"]
    )
    assert "⛔ refused" in out and "protected CI files" in out


def test_warning_banners_the_report():
    out = RP.render(
        "1",
        CELLS,
        [_res("gemma2.apc.off.single")],
        notes=["WARNING: this pull request modifies CI harness files (ci/report.py)"],
    )
    assert "⚠️" in out and "modifies CI harness files" in out
    # the benchmark still ran, so the normal detail is present too
    assert "per-cell detail" in out


def test_status_for_maps_headline_to_commit_state():
    from ci.report import status_for

    assert status_for("**mlx-vlm benchmark** — no regression")[0] == "success"
    assert status_for("**mlx-vlm benchmark** — 2 regression(s)")[0] == "failure"
    assert status_for("**mlx-vlm benchmark** — ⛔ refused: x")[0] == "failure"
    assert status_for("**mlx-vlm benchmark** — running — 1 of 4 done")[0] == "pending"
    assert (
        status_for("**mlx-vlm benchmark** — all devices busy — re-run")[0] == "success"
    )
