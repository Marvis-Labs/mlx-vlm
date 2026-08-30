from pathlib import Path

import yaml

from ci.change_rules import ChangeContext, ChangeMatch
from ci.delegator import ModelPath
from ci.mlp_change import MLPChange

CLASSES = """
class SwiGLUMLP:
    def __call__(self, x):
        return x

class DeepseekMLP:
    def __call__(self, x):
        return x

class GELUMLP:
    def __call__(self, x):
        return x

class FastGELUMLP:
    def __call__(self, x):
        return x

class TanhGELUMLP:
    def __call__(self, x):
        return x
"""


class FakeSource:
    def __init__(self, values):
        self.values = values

    def read_text(self, revision, path):
        return self.values[(revision, path)]


def component(tmp_path: Path, source: FakeSource) -> MLPChange:
    model = {
        "schema_version": 1,
        "synthetic_profiles": {"dense": {}},
        "models": {
            "swi": {
                "capabilities": ["vlm"],
                "synthetic": {"status": "TODO"},
                "hf_checkpoint": {
                    "status": "configured",
                    "repo": "example/swi",
                    "revision": "abc",
                    "expected_model_type": "swi",
                    "weight": {"bytes": 1024},
                },
            }
        },
    }
    scenario = {
        "schema_version": 1,
        "defaults_by_capability": {"vlm": "animal"},
        "scenarios": {"animal": {}},
    }
    manifest = {
        "schema_version": 1,
        "source": "mlx_vlm/models/mlp.py",
        "symbols": {
            "SwiGLUMLP": {"expected_families": 1, "families": ["swi"]},
            "DeepseekMLP": {"expected_families": 1, "families": ["deep"]},
            "GELUMLP": {"expected_families": 1, "families": ["gelu"]},
            "FastGELUMLP": {"expected_families": 1, "families": ["fast"]},
            "TanhGELUMLP": {"expected_families": 1, "families": ["tanh"]},
        },
    }
    model_path = tmp_path / "model.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    manifest_path = tmp_path / "mlp.yaml"
    model_path.write_text(yaml.safe_dump(model))
    scenario_path.write_text(yaml.safe_dump(scenario))
    manifest_path.write_text(yaml.safe_dump(manifest))
    return MLPChange(manifest_path, ModelPath(model_path, scenario_path), source)


def context(source_values, base_source=CLASSES, head_source=CLASSES):
    files = [
        "mlx_vlm/models/mlp.py",
        "mlx_vlm/models/swi/model.py",
        "mlx_vlm/models/deep/model.py",
        "mlx_vlm/models/gelu/model.py",
        "mlx_vlm/models/fast/model.py",
        "mlx_vlm/models/tanh/model.py",
    ]
    imports = {
        "swi": "SwiGLUMLP",
        "deep": "DeepseekMLP",
        "gelu": "GELUMLP",
        "fast": "FastGELUMLP",
        "tanh": "TanhGELUMLP",
    }
    source_values[("base", "mlx_vlm/models/mlp.py")] = base_source
    source_values[("head", "mlx_vlm/models/mlp.py")] = head_source
    for revision in ("base", "head"):
        for family, symbol in imports.items():
            source_values[(revision, f"mlx_vlm/models/{family}/model.py")] = (
                f"from ..mlp import {symbol}\n"
            )
    return ChangeContext.create(
        ["mlx_vlm/models/mlp.py"],
        files,
        files,
        base_sha="base",
        head_sha="head",
        tree_state_known=True,
    )


def match():
    return ChangeMatch("mlp_change", "mlp_change", "mlx_vlm/models/mlp.py", {})


def test_changed_class_expands_only_its_pinned_consumers(tmp_path):
    values = {}
    head = CLASSES.replace(
        "return x\n\nclass GELUMLP", "return x + 1\n\nclass GELUMLP", 1
    )
    planner = component(tmp_path, FakeSource(values))

    plan = planner.plan([match()], context(values, head_source=head))

    assert plan["blocked"] == []
    assert [job["model"] for job in plan["jobs"]] == ["deep"]
    assert plan["jobs"][0]["phases"] == ["mlp_contract"]
    assert plan["jobs"][0]["unavailable_phases"] == {
        "hf_checkpoint": "missing_model_config"
    }


def test_configured_checkpoint_is_added_without_full_model_synthetic_adapter(tmp_path):
    values = {}
    head = CLASSES.replace(
        "return x\n\nclass DeepseekMLP", "return x + 1\n\nclass DeepseekMLP", 1
    )
    planner = component(tmp_path, FakeSource(values))

    plan = planner.plan([match()], context(values, head_source=head))

    job = plan["jobs"][0]
    assert job["model"] == "swi"
    assert job["phases"] == ["mlp_contract", "hf_checkpoint"]
    assert job["hf_checkpoint"]["repo"] == "example/swi"
    assert "synthetic" not in job


def test_non_class_semantic_change_conservatively_selects_all_symbols(tmp_path):
    values = {}
    planner = component(tmp_path, FakeSource(values))

    plan = planner.plan([match()], context(values, head_source="VALUE = 1\n" + CLASSES))

    assert len(plan["jobs"]) == 5
    assert plan["metadata"]["detection"] == "conservative_module_change"


def test_comments_only_change_emits_no_jobs(tmp_path):
    values = {}
    planner = component(tmp_path, FakeSource(values))

    plan = planner.plan(
        [match()], context(values, head_source="# formatting only\n" + CLASSES)
    )

    assert plan["jobs"] == []
    assert plan["metadata"]["detection"] == "no_semantic_change"


def test_removed_consumer_is_still_discovered_from_base_tree(tmp_path):
    values = {}
    ctx = context(values)
    values[("head", "mlx_vlm/models/deep/model.py")] = "VALUE = 1\n"
    head = CLASSES.replace(
        "return x\n\nclass GELUMLP", "return x + 1\n\nclass GELUMLP", 1
    )
    values[("head", "mlx_vlm/models/mlp.py")] = head
    planner = component(tmp_path, FakeSource(values))

    plan = planner.plan([match()], ctx)

    assert plan["blocked"] == []
    assert [job["model"] for job in plan["jobs"]] == ["deep"]
