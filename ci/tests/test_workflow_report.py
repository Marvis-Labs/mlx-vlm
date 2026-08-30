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
        "123",
        "abc123",
        "success",
    )

    assert result["kind"] == "ci_execution"
    assert result["outcome"] == "passed"
    assert result["run_url"] == "https://example.com/run"


def test_exhausted_dispatch_becomes_no_runner_result():
    result = report(
        plan(),
        dispatch("no_eligible_runner"),
        None,
        "run",
        "123",
        "abc123",
        "skipped",
    )
    assert result["outcome"] == "no_eligible_runner"


def test_missing_runner_artifact_is_infrastructure_failure():
    result = report(plan(), dispatch(), None, "run", "456", "abc123", "failure")

    execution = result["results"][0]
    assert result["attempt_id"] == "456"
    assert execution["outcome"] == "infrastructure_failure"
    assert execution["selected_device"] is None
    assert execution["findings"]["execution_status"] == "failure"


def test_cancelled_runner_attempt_is_reported_as_cancelled():
    result = report(plan(), dispatch(), None, "run", "789", "abc123", "cancelled")
    assert result["results"][0]["outcome"] == "cancelled"


def test_prepare_failure_builds_renderable_fallback():
    result = report(None, None, None, "run", "999", "new-head", "skipped")

    assert result["head_sha"] == "new-head"
    assert result["results"] == []
    assert result["errors"][0]["code"] == "attempt_preparation_failed"
