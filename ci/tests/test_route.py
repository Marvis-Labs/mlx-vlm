"""The router's decisions, pinned so they cannot silently drift.

Pure and offline: no hardware, no network. Each test names one of the routing
behaviours the CI depends on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ci import route as R  # noqa: E402


def _first(paths, **kw):
    return R.route(paths, **kw)


def test_model_path_selects_that_arch_only():
    r = R.route(["mlx_vlm/models/gemma2/language.py"])
    assert r["cells"], "a model change must produce cells"
    assert {c["arch"] for c in r["cells"]} == {"gemma2"}


def test_test_and_doc_files_do_not_trigger():
    for p in (
        "mlx_vlm/models/gemma2/test_gemma2.py",
        "mlx_vlm/models/gemma2/README.md",
    ):
        assert R.route([p])["cells"] == [], f"{p} should not benchmark"


def test_component_change_fans_out_by_signature():
    r = R.route(["mlx_vlm/models/cache.py"])
    archs = {c["arch"] for c in r["cells"]}
    assert len(archs) > 1, "a component change spans many architectures"


def test_unknown_model_is_noted_not_crashed():
    r = R.route(["mlx_vlm/models/ghost/language.py"])
    assert r["cells"] == [] and r["notes"], "unknown arch -> note, no cells"


def test_unrouted_path_is_reported():
    r = R.route(["setup.py"])
    assert any("unrouted" in n for n in r["notes"])


def test_dedup_across_model_and_component():
    r = R.route(["mlx_vlm/models/gemma2/language.py", "mlx_vlm/models/cache.py"])
    ids = [c["id"] for c in r["cells"]]
    assert len(ids) == len(set(ids)), "cells must be unique across classes"


def test_every_cell_is_well_formed():
    r = R.route(["mlx_vlm/models/cache.py"])
    for c in r["cells"]:
        assert c["variants"], "a cell must carry precision variants"
        assert c["min_gb"] > 0, "min_gb must be positive"
        assert c["runs_on"][-1].startswith("mem-"), "cell must have a tier label"
        assert all(v["requires_gb"] > 0 for v in c["variants"])


def test_cell_labelled_by_smallest_variant():
    # gemma2: 4-bit is smallest, fits mem-16; label must be mem-16 not the bf16 tier
    r = R.route(["mlx_vlm/models/gemma2/language.py"])
    assert r["cells"][0]["runs_on"][-1] == "mem-16"


def test_empty_diff_is_empty():
    assert R.route([])["cells"] == []


def test_new_model_undeclared_asks_for_declaration():
    r = R.route(["mlx_vlm/models/gliner2_5/language.py"])
    assert any("parity_models.yaml" in n for n in r["notes"])
    assert r["cells"] == []


def test_declared_new_model_routes_to_the_parity_gate(tmp_path, monkeypatch):
    # A model absent from the matrix has no previous revision to compare, so if
    # it is declared in parity_models.yaml it must go to the correctness gate:
    # exactly one parity cell, never the performance path. This guards the
    # second gate from silently going dead -- which is what happens once every
    # declared model has been sized into the matrix and reads as "known".
    decl = tmp_path / "parity_models.yaml"
    decl.write_text(
        "zzparitytest:\n  mlx: org/zzparitytest-bf16\n  ref: org/zzparitytest\n"
    )
    monkeypatch.setattr(R, "PARITY_MODELS", decl)
    r = R.route(["mlx_vlm/models/zzparitytest/language.py"])
    ids = [c["id"] for c in r["cells"]]
    assert ids == ["zzparitytest.parity"], ids
    cell = r["cells"][0]
    assert cell["runs_on"][-1] == "parity", cell["runs_on"]
    assert cell["component"] == "parity"
    assert "greedy_agreement" in cell["metrics"]
    assert not any(c["component"] != "parity" for c in r["cells"])
