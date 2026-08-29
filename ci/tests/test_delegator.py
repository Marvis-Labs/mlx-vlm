import json
from pathlib import Path

import yaml

from ci.delegator import Delegator, ModelPath, _parse_name_status, main


def write_configs(tmp_path: Path) -> tuple[Path, Path]:
    model_config = {
        "schema_version": 1,
        "synthetic_profiles": {"dense_vlm": {}},
        "models": {
            "ready": {
                "capabilities": ["vision_language"],
                "synthetic": {
                    "status": "configured",
                    "adapter": "ready",
                    "profile": "dense_vlm",
                },
                "hf_checkpoint": {
                    "status": "configured",
                    "repo": "example/ready",
                    "revision": "abc123",
                    "expected_model_type": "ready",
                    "weight": {"bytes": 1024, "gib": 0.01},
                },
            },
            "todo": {
                "synthetic": {"status": "TODO"},
                "hf_checkpoint": {"status": "TODO"},
            },
        },
    }
    scenario_config = {
        "schema_version": 1,
        "defaults_by_capability": {"vision_language": "vlm_animal"},
        "scenarios": {"vlm_animal": {}},
    }
    model_path = tmp_path / "model_path.yaml"
    scenario_path = tmp_path / "model-path-scenario.yaml"
    model_path.write_text(yaml.safe_dump(model_config))
    scenario_path.write_text(yaml.safe_dump(scenario_config))
    return model_path, scenario_path


def make_delegator(tmp_path: Path) -> Delegator:
    model_config, scenario_config = write_configs(tmp_path)
    return Delegator([ModelPath(model_config, scenario_config)])


def test_configured_model_emits_synthetic_and_checkpoint_jobs(tmp_path):
    plan = make_delegator(tmp_path).plan(
        [
            "mlx_vlm/models/ready/vision.py",
            "mlx_vlm/models/ready/config.py",
            "mlx_vlm/models/ready/config.py",
        ]
    )

    assert plan["components"] == ["model_path"]
    assert [job["mode"] for job in plan["jobs"]] == [
        "synthetic",
        "hf_checkpoint",
    ]
    assert plan["jobs"][0]["changed_paths"] == [
        "mlx_vlm/models/ready/config.py",
        "mlx_vlm/models/ready/vision.py",
    ]
    assert plan["jobs"][0]["scenarios"] == ["vlm_animal"]
    assert plan["jobs"][1]["hf_checkpoint"]["weight"]["bytes"] == 1024
    assert plan["blocked"] == []


def test_todo_model_is_blocked_for_both_modes(tmp_path):
    plan = make_delegator(tmp_path).plan(["mlx_vlm/models/todo/model.py"])

    assert plan["jobs"] == []
    assert [(item["mode"], item["reason"]) for item in plan["blocked"]] == [
        ("synthetic", "not_configured"),
        ("hf_checkpoint", "not_configured"),
    ]


def test_unconfigured_new_model_is_not_silently_skipped(tmp_path):
    plan = make_delegator(tmp_path).plan(["mlx_vlm/models/new_model/model.py"])

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "missing_model_config"


def test_invalid_scenario_config_blocks_configured_modes(tmp_path):
    delegator = make_delegator(tmp_path)
    delegator.components[0].models["ready"]["scenarios"] = [{"invalid": True}]

    plan = delegator.plan(["mlx_vlm/models/ready/model.py"])

    assert plan["jobs"] == []
    assert [(item["mode"], item["reason"]) for item in plan["blocked"]] == [
        ("synthetic", "invalid_scenarios"),
        ("hf_checkpoint", "invalid_scenarios"),
    ]


def test_shared_model_component_and_unrelated_files_are_ignored(tmp_path):
    plan = make_delegator(tmp_path).plan(
        ["mlx_vlm/models/attention.py", "mlx_vlm/server/app.py", "README.md"]
    )

    assert plan == {
        "schema_version": 1,
        "components": [],
        "jobs": [],
        "blocked": [],
    }


def test_rename_includes_old_and_new_paths():
    output = (
        b"R100\0mlx_vlm/models/old/model.py\0"
        b"mlx_vlm/models/new/model.py\0M\0README.md\0"
    )

    assert _parse_name_status(output) == (
        "mlx_vlm/models/old/model.py",
        "mlx_vlm/models/new/model.py",
        "README.md",
    )


def test_cli_writes_json_plan(tmp_path, monkeypatch):
    model_config, scenario_config = write_configs(tmp_path)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(
        "ci.delegator.default_delegator",
        lambda: Delegator([ModelPath(model_config, scenario_config)]),
    )

    assert (
        main(
            [
                "--changed-file",
                "mlx_vlm/models/ready/model.py",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads(output.read_text())
    assert len(plan["jobs"]) == 2
    assert plan["blocked"] == []


def test_repository_configured_models_are_routable():
    config_directory = Path(__file__).parents[1]
    model_path = ModelPath(
        config_directory / "model_path.yaml",
        config_directory / "model-path-scenario.yaml",
    )
    configured = [
        name
        for name, model in model_path.models.items()
        if model.get("synthetic", {}).get("status") == "configured"
        and model.get("hf_checkpoint", {}).get("status") == "configured"
    ]

    plan = Delegator([model_path]).plan(
        [f"mlx_vlm/models/{name}/config.py" for name in configured]
    )

    assert len(configured) == 30
    assert len(plan["jobs"]) == 60
    assert plan["blocked"] == []
