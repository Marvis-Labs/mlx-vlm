import json
from pathlib import Path

import pytest

from ci.control import (
    ControlError,
    ExecutionOutcome,
    PlanningOutcome,
    configuration_digest,
    control_record,
    execution_outcome,
    main,
    planning_outcome,
    release_control,
    render_status,
)


def plan(*, jobs=None, gates=None, blocked=None, head_sha="abc123"):
    return {
        "schema_version": 1,
        "head_sha": head_sha,
        "rules": [],
        "components": [],
        "jobs": jobs or [],
        "gates": gates or [],
        "blocked": blocked or [],
    }


def pending_job(mode):
    return {
        "id": f"new_model_path:example:{mode}",
        "component": "new_model_path",
        "model": "example",
        "mode": mode,
    }


def approval_gate():
    configuration = {
        "synthetic": {"adapter": "example", "profile": "dense_vlm"},
        "hf_checkpoint": {
            "repo": "example/model",
            "revision": "revision",
            "expected_model_type": "example",
            "weight": {"bytes": 1024},
        },
        "scenarios": ["vlm_animal"],
    }
    return {
        "id": "new_model_path:example:abc123",
        "type": "maintainer_approval",
        "status": "awaiting_maintainer_approval",
        "component": "new_model_path",
        "model": "example",
        "head_sha": "abc123",
        "configuration_digest": configuration_digest(configuration),
        "changed_paths": ["mlx_vlm/models/example/model.py"],
        "requested_jobs": ["synthetic", "hf_checkpoint"],
        "configuration": configuration,
        "pending_jobs": [pending_job("synthetic"), pending_job("hf_checkpoint")],
    }


def test_planning_outcome_precedence():
    assert planning_outcome(plan()) is PlanningOutcome.READY
    assert (
        planning_outcome(plan(gates=[approval_gate()]))
        is PlanningOutcome.AWAITING_APPROVAL
    )
    assert (
        planning_outcome(
            plan(gates=[approval_gate()], blocked=[{"reason": "missing_config"}])
        )
        is PlanningOutcome.BLOCKED
    )


def test_control_record_normalizes_blockers():
    record = control_record(
        plan(
            blocked=[
                {
                    "component": "new_model_path",
                    "model": "example",
                    "reason": "model_manifest_not_updated",
                    "changed_paths": ["mlx_vlm/models/example/model.py"],
                }
            ]
        ),
        "Marvis-Labs/mlx-vlm-ci",
        8,
    )

    assert record["outcome"] == "blocked"
    assert record["errors"] == [
        {
            "code": "model_manifest_not_updated",
            "category": "configuration",
            "component": "new_model_path",
            "subject": "example",
            "retryable": False,
            "user_actionable": True,
            "details": {"changed_paths": ["mlx_vlm/models/example/model.py"]},
        }
    ]


def test_release_turns_approved_gate_into_jobs():
    record = control_record(plan(gates=[approval_gate()]), "Marvis-Labs/mlx-vlm-ci", 8)

    released = release_control(record, "abc123")

    assert released["kind"] == "approved_job_plan"
    assert released["outcome"] == "ready"
    assert [job["mode"] for job in released["jobs"]] == [
        "synthetic",
        "hf_checkpoint",
    ]
    assert released["gates"][0]["status"] == "approved"
    assert released["approval"] == {
        "mechanism": "github_environment",
        "head_sha": "abc123",
        "gate_ids": ["new_model_path:example:abc123"],
    }
    assert released["job_plan_digest"].startswith("sha256:")


def test_release_rejects_stale_head():
    record = control_record(plan(gates=[approval_gate()]), "Marvis-Labs/mlx-vlm-ci", 8)

    with pytest.raises(ControlError, match="head changed"):
        release_control(record, "different")


def test_release_rejects_tampered_configuration():
    gate = approval_gate()
    gate["configuration"]["hf_checkpoint"]["repo"] = "attacker/model"
    record = control_record(plan(gates=[gate]), "Marvis-Labs/mlx-vlm-ci", 8)

    with pytest.raises(ControlError, match="digest does not match"):
        release_control(record, "abc123")


def test_release_rejects_duplicate_job_ids():
    duplicate = pending_job("synthetic")
    gate = approval_gate()
    gate["pending_jobs"] = [duplicate, duplicate]
    record = control_record(plan(gates=[gate]), "Marvis-Labs/mlx-vlm-ci", 8)

    with pytest.raises(ControlError, match="job ids must be unique"):
        release_control(record, "abc123")


def test_release_rejects_job_outside_gate_scope():
    gate = approval_gate()
    gate["pending_jobs"][0]["model"] = "different"
    record = control_record(plan(gates=[gate]), "Marvis-Labs/mlx-vlm-ci", 8)

    with pytest.raises(ControlError, match="exceed its scope"):
        release_control(record, "abc123")


def test_status_renderer_is_centralized_and_suppresses_mentions():
    gate = approval_gate()
    gate["model"] = "@reviewer|model"
    record = control_record(
        plan(gates=[gate]), "Marvis-Labs/mlx-vlm-ci", 8, "https://example.com/run"
    )

    rendered = render_status(record)

    assert rendered.startswith("<!-- mlx-vlm-ci:plan -->")
    assert "Awaiting maintainer approval" in rendered
    assert "@\u200breviewer\\|model" in rendered
    assert "No Apple Silicon job starts" in rendered


@pytest.mark.parametrize(
    ("exit_code", "cancelled", "infrastructure_failure", "expected"),
    [
        (0, False, False, ExecutionOutcome.PASSED),
        (1, False, False, ExecutionOutcome.TEST_FAILURE),
        (None, False, False, ExecutionOutcome.INFRASTRUCTURE_FAILURE),
        (1, False, True, ExecutionOutcome.INFRASTRUCTURE_FAILURE),
        (1, True, False, ExecutionOutcome.CANCELLED),
    ],
)
def test_execution_outcomes(exit_code, cancelled, infrastructure_failure, expected):
    assert (
        execution_outcome(
            exit_code,
            cancelled=cancelled,
            infrastructure_failure=infrastructure_failure,
        )
        is expected
    )


def test_release_cli_writes_runner_manifest(tmp_path):
    record = control_record(plan(gates=[approval_gate()]), "Marvis-Labs/mlx-vlm-ci", 8)
    source = tmp_path / "control.json"
    output = tmp_path / "runner-plan.json"
    markdown = tmp_path / "summary.md"
    source.write_text(json.dumps(record))

    assert (
        main(
            [
                "release",
                "--control",
                str(source),
                "--current-head",
                "abc123",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["kind"] == "approved_job_plan"
    assert "ready for runner dispatch" in markdown.read_text()


def test_plan_cli_uses_immutable_repository_head(tmp_path):
    repository_root = Path(__file__).parents[2]
    output = tmp_path / "control.json"
    markdown = tmp_path / "summary.md"

    assert (
        main(
            [
                "plan",
                "--base",
                "HEAD",
                "--head",
                "HEAD",
                "--repository",
                "Marvis-Labs/mlx-vlm-ci",
                "--repository-path",
                str(repository_root),
                "--pr",
                "8",
                "--rules-config",
                str(repository_root / "ci/change-rules.yaml"),
                "--model-config",
                str(repository_root / "ci/model_path.yaml"),
                "--scenario-config",
                str(repository_root / "ci/model-path-scenario.yaml"),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    record = json.loads(output.read_text())
    assert record["outcome"] == "ready"
    assert record["jobs"] == []
    assert record["head_sha"]
