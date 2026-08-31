from ci.worker_result import finalize


def job():
    return {
        "id": "model_path:qwen2_vl:hf_checkpoint",
        "component": "model_path",
        "model": "qwen2_vl",
        "mode": "hf_checkpoint",
    }


def test_missing_runner_result_is_infrastructure_failure():
    work = job()
    work["profile"] = "dense"
    result = finalize(work, None)
    assert result["outcome"] == "infrastructure_failure"
    assert result["profile"] == "dense"
    assert result["job_id"] == "model_path:qwen2_vl:hf_checkpoint"


def test_findings_verdict_and_metrics_are_promoted():
    result = finalize(
        job(),
        {
            "decision": "accepted",
            "outcome": "passed",
            "findings": {
                "verdict": "regressed",
                "metrics": {"ttft_ms": {"base": 10, "head": 20}},
            },
        },
    )

    assert result["outcome"] == "regressed"
    assert "ttft_ms" in result["metrics"]
    assert result["component"] == "model_path"
    assert result["job_id"] == "model_path:qwen2_vl:hf_checkpoint"
