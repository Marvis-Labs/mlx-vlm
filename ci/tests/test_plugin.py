import hashlib
from pathlib import Path

import pytest
import yaml
from mlx_ci.repository.components import ExecutionContext

from ci import plugin as registry

ROOT = Path(__file__).parents[2]


def test_component_registrations_are_unique_and_self_contained():
    names = [registration.name for registration in registry.REGISTRATIONS]

    assert len(names) == len(set(names))
    assert registry.contributor_config_paths() == (
        "config/models.yaml",
        "config/scenarios.yaml",
    )


def test_removing_a_registration_removes_its_work_and_phases(monkeypatch):
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "model_path"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    context = ExecutionContext(
        Path("job.json"),
        Path("control"),
        Path("base"),
        Path("head"),
    )
    assert "synthetic" not in registry.phase_commands(context)
    with pytest.raises(ValueError, match="unregistered work item"):
        registry.validate_job(
            {
                "work_type": "ModelPath",
                "component": "model_path",
                "phases": ["synthetic"],
            }
        )


def test_removing_docs_registration_removes_its_planner(monkeypatch):
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "docs_change"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    assert all(
        planner.name != "docs_change"
        for planner in registry.planners(Path("ci"), Path("."))
    )


def test_registered_phases_build_commands_without_executor_switches():
    context = ExecutionContext(
        Path("job.json"),
        Path("control"),
        Path("base"),
        Path("head"),
    )

    commands = registry.phase_commands(context)

    assert set(commands) == {"synthetic", "hf_checkpoint"}
    assert commands["synthetic"][1].endswith("ci/model_path/synthetic_compare.py")
    assert "--scenarios" in commands["hf_checkpoint"]


def test_workflow_calls_only_generic_component_entry_points():
    workflow = Path(__file__).parents[2] / ".github" / "workflows" / "bench.yml"
    source = workflow.read_text()

    assert "repos/Marvis-Labs/mlx-ci/dispatches" in source
    assert not (Path(__file__).parents[1] / "control.py").exists()
    assert not (Path(__file__).parents[1] / "work_executor.py").exists()
    assert not (Path(__file__).parents[1] / "report.py").exists()
    assert "model_path_work.py" not in source
    assert "model_path_compare.py" not in source
    assert "kv_cache_contract_compare.py" not in source


def test_gate_validation_is_owned_by_the_registered_component(monkeypatch):
    gate = {
        "component": "new_model_path",
        "model": "new_model",
        "requested_phases": ["synthetic"],
        "pending_work": {
            "work_type": "ModelPath",
            "component": "model_path",
            "model": "new_model",
            "phases": ["synthetic"],
        },
    }
    registry.validate_gate(gate)
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "model_path"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    with pytest.raises(ValueError, match="no unique gate validator"):
        registry.validate_gate(gate)


def test_acceptance_fixture_is_pinned_and_consistent():
    fixture = yaml.safe_load((ROOT / "ci/config/acceptance.yaml").read_text())
    profiles = yaml.safe_load((ROOT / "ci/config/models.yaml").read_text())
    model = profiles["models"][fixture["model"]]
    checkpoint = fixture["hf_checkpoint"]

    assert fixture["phases"] == ["synthetic", "hf_checkpoint"]
    assert model["synthetic"]["adapter"] == fixture["synthetic"]["adapter"]
    assert model["synthetic"]["profile"] == fixture["synthetic"]["profile"]
    assert model["hf_checkpoint"]["repo"] == checkpoint["repo"]
    assert model["hf_checkpoint"]["revision"] == checkpoint["revision"]
    assert model["hf_checkpoint"]["weight"]["bytes"] == checkpoint["weight_bytes"]
    assert fixture["input"]["max_tokens"] == 16
    assert fixture["thresholds"]["performance_percent"] == 5.0
    asset = ROOT / fixture["input"]["asset"]
    assert (
        hashlib.sha256(asset.read_bytes()).hexdigest()
        == fixture["input"]["asset_sha256"]
    )
