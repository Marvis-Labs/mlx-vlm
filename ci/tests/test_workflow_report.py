from ci.workflow_report import report, report_batch, report_coalesced


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
            "id": "model_path:qwen2_vl",
            "work_type": "ModelPath",
            "component": "model_path",
            "model": "qwen2_vl",
            "phases": ["synthetic", "hf_checkpoint"],
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
    lease = {"attempt_id": "123", "device": "mini"}
    selected_dispatch = dispatch()
    selected_dispatch["lease"] = lease
    result = report(
        plan(),
        selected_dispatch,
        {"outcome": "passed", "component": "model_path"},
        "https://example.com/run",
        "123",
        "abc123",
        "success",
    )

    assert result["kind"] == "ci_execution"
    assert result["outcome"] == "passed"
    assert result["run_url"] == "https://example.com/run"
    assert result["device_leases"] == [lease]


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


def test_blocked_plan_remains_a_planning_verdict():
    value = plan()
    value.update(
        {
            "kind": "ci_control",
            "outcome": "blocked",
            "errors": [
                {
                    "code": "security_policy_violation",
                    "component": "security",
                    "subject": "pull_request",
                }
            ],
        }
    )

    result = report(value, None, None, "run", "blocked-1", "abc123", "skipped")

    assert result["kind"] == "ci_execution"
    assert result["outcome"] == "blocked"
    assert result["attempt_id"] == "blocked-1"
    assert result["results"] == []
    assert result["errors"] == value["errors"]


def test_batch_reports_each_model_independently():
    first = dispatch()
    first["job"]["model"] = "first"
    first["job"]["id"] = "model_path:first"
    second = dispatch()
    second["job"]["model"] = "second"
    second["job"]["id"] = "model_path:second"
    batch = {
        "items": [
            {"key": "first", "work": first["job"], "dispatch": first, "lease": {}},
            {"key": "second", "work": second["job"], "dispatch": second, "lease": {}},
        ]
    }

    result = report_batch(
        plan(),
        batch,
        {
            "first": {"component": "model_path", "model": "first", "outcome": "passed"},
            "second": {
                "component": "model_path",
                "model": "second",
                "outcome": "regressed",
            },
        },
        "run",
        "123",
        "abc123",
        "success",
    )

    assert [item["model"] for item in result["results"]] == ["first", "second"]
    assert result["outcome"] == "regressed"


def test_final_exhausted_dispatch_overrides_prior_decline_result():
    exhausted = dispatch("no_eligible_runner")
    exhausted["attempts"] = [
        {
            "device": "mini",
            "label": "device-mini",
            "memory_gib": 16,
            "decision": "declined",
            "reason": "declined_memory",
            "details": {},
        }
    ]
    batch = {
        "items": [
            {
                "key": "qwen",
                "work": exhausted["job"],
                "dispatch": exhausted,
                "lease": None,
            }
        ]
    }

    result = report_batch(
        plan(),
        batch,
        {"qwen": {"decision": "declined", "outcome": "declined", "device": "mini"}},
        "run",
        "123",
        "abc123",
        "success",
    )

    assert result["outcome"] == "no_eligible_runner"


def test_coalesced_attempt_is_immutable_terminal_record():
    value = plan()
    value["jobs"] = [dispatch()["job"]]

    result = report_coalesced(value, "run-2", "two", "abc123", "one", "run-1")

    assert result["outcome"] == "coalesced"
    assert result["attempt_id"] == "two"
    assert result["coalesced_with"] == {"attempt_id": "one", "run_url": "run-1"}
