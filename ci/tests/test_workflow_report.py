from ci.workflow_report import report


def plan():
    return {
        "schema_version": 1,
        "kind": "approved_job_plan",
        "head_sha": "abc123",
        "outcome": "ready",
        "jobs": [],
        "gates": [],
        "errors": [],
    }


def dispatch(outcome="dispatching"):
    return {
        "schema_version": 1,
        "kind": "device_dispatch",
        "job": {
            "id": "model_path:qwen2_vl:hf_checkpoint",
            "component": "model_path",
            "model": "qwen2_vl",
            "mode": "hf_checkpoint",
            "required_memory_gib": 8,
            "required_disk_gib": 4,
        },
        "required_memory_gib": 8,
        "required_disk_gib": 4,
        "candidates": [],
        "unavailable": [],
        "attempts": [],
        "outcome": outcome,
        "next_device": None,
        "selected_device": None,
    }


def test_execution_result_is_attached_to_approved_plan():
    result = report(
        plan(),
        dispatch(),
        {"outcome": "passed", "component": "model_path"},
        "https://example.com/run",
    )

    assert result["kind"] == "ci_execution"
    assert result["outcome"] == "passed"
    assert result["run_url"] == "https://example.com/run"


def test_exhausted_dispatch_becomes_no_runner_result():
    result = report(plan(), dispatch("no_eligible_runner"), None, "run")
    assert result["outcome"] == "no_eligible_runner"
