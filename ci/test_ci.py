import hashlib
import json
import subprocess
from pathlib import Path

from ci import plan_ci, render_comment
from ci.execution_security import ExecutionSecurityError, validate_job
from ci.output import OutputError, render_coalesced, validate_bundle, validate_dispatch
from ci.work_executor import _server_contract

CATALOG = json.loads(
    (Path(__file__).parents[1] / "mlx_vlm/tests/model_cases.json").read_text()
)


def test_model_path_planning():
    plan = plan_ci(
        [
            "mlx_vlm/models/qwen2_vl/vision.py",
            "mlx_vlm/models/qwen2_vl/language.py",
            "mlx_vlm/models/florence2/florence2.py",
        ],
        CATALOG,
    )
    assert [job["subject"] for job in plan["jobs"]] == ["florence2", "qwen2_vl"]
    assert all(job["phases"] == ["synthetic", "checkpoint"] for job in plan["jobs"])
    assert all(job["work"]["checkpoint"]["profile"] == "image" for job in plan["jobs"])
    assert all(
        job["work"]["checkpoint"]["asset"] == "image/image.json" for job in plan["jobs"]
    )


def test_model_path_without_checkpoint_and_missing_case():
    synthetic = plan_ci(["mlx_vlm/models/aya_vision/aya_vision.py"], CATALOG)
    assert synthetic["jobs"][0]["phases"] == ["synthetic"]
    assert synthetic["jobs"][0]["artifact"] is None
    blocked = plan_ci(["mlx_vlm/models/new_family/model.py"], CATALOG)
    assert blocked["jobs"] == []
    assert blocked["blocked"][0]["reason"] == "model_case_missing"


def test_glm5_next_model_path_uses_its_model_contract():
    plan = plan_ci(["mlx_vlm/models/glm5_next/language.py"], CATALOG)
    assert plan["blocked"] == []
    assert plan["jobs"][0]["work"]["synthetic"]["selectors"] == [
        "mlx_vlm/tests/test_models.py::test_model_contract[TestModels.glm5_next]"
    ]


def test_server_change_is_one_independent_job():
    plan = plan_ci(
        [
            "mlx_vlm/models/qwen2_vl/vision.py",
            "mlx_vlm/server/openai.py",
            "mlx_vlm/server/embeddings.py",
        ],
        CATALOG,
    )
    server = [job for job in plan["jobs"] if job["component"] == "server_change"]
    assert len(server) == 1
    assert server[0]["work"]["server_contract"]["profiles"] == [
        "embeddings",
        "openai",
    ]
    assert server[0]["work"]["server_contract"]["selectors"] == [
        "mlx_vlm/tests/test_server.py"
    ]


def test_server_test_change_selects_all_profiles():
    plan = plan_ci(["mlx_vlm/tests/test_server.py"], CATALOG)
    assert plan["jobs"][0]["component"] == "server_change"
    assert plan["jobs"][0]["work"]["server_contract"]["profiles"] == ["all"]


def test_server_contract_uses_main_as_context_and_head_as_verdict(monkeypatch):
    job = plan_ci(["mlx_vlm/server/openai.py"], CATALOG)["jobs"][0]
    outcomes = iter(
        [
            subprocess.CompletedProcess([], 1, "1 failed in 1.0s\n", ""),
            subprocess.CompletedProcess(
                [], 0, "12 passed in 2.0s\n", "trailing warning\n"
            ),
        ]
    )
    monkeypatch.setattr("ci.work_executor._run", lambda *_: next(outcomes))
    result = _server_contract(job, Path("control"), Path("base"), Path("head"))
    assert result["status"] == "passed"
    assert result["detail"] == (
        "Main: 1 failed in 1.0s; PR: 12 passed in 2.0s; profiles: openai"
    )


def test_server_contract_fails_when_head_fails(monkeypatch):
    job = plan_ci(["mlx_vlm/server/openai.py"], CATALOG)["jobs"][0]
    outcomes = iter(
        [
            subprocess.CompletedProcess([], 0, "12 passed in 2.0s\n", ""),
            subprocess.CompletedProcess([], 1, "1 failed in 1.0s\n", ""),
        ]
    )
    monkeypatch.setattr("ci.work_executor._run", lambda *_: next(outcomes))
    result = _server_contract(job, Path("control"), Path("base"), Path("head"))
    assert result["status"] == "failed"


def test_output_hides_runner_identity_and_bolds_four_percent():
    attempt = {
        "run_id": 1842,
        "run_attempt": 1,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "changed_files": ["mlx_vlm/server/openai.py"],
    }
    job = plan_ci(attempt["changed_files"], CATALOG)["jobs"][0]
    job.update(manifest_digest="c" * 64)
    result = {
        "schema_version": 2,
        "job_id": "server-change",
        "manifest_digest": "c" * 64,
        "status": "passed",
        "device": {"chip": "Apple M4", "memory_gib": 16},
        "cache": "not_applicable",
        "duration_ms": 1250,
        "checks": [
            {
                "name": "Server contract",
                "category": "correctness",
                "status": "passed",
                "detail": "All endpoint tests passed",
            }
        ],
        "metrics": [
            {
                "name": "TTFT",
                "unit": "ms",
                "base": 100,
                "head": 104,
                "change_pct": 4.0,
                "verdict": "regressed",
            },
            {
                "name": "Wall time",
                "unit": "ms",
                "base": 100,
                "head": 103.99,
                "change_pct": 3.99,
                "verdict": "stable",
            },
        ],
    }
    rendered = render_comment(
        attempt,
        {"jobs": [job], "blocked": []},
        [result],
        "https://github.com/Marvis-Labs/mlx-ci/actions/runs/1842",
    )
    assert "Apple M4 · 16 GB unified memory" in rendered
    assert "### Mixie" in rendered
    assert "runner" not in rendered.lower()
    assert "**+4.00%**" in rendered
    assert "+3.99%" in rendered and "**+3.99%**" not in rendered
    assert not any(character in rendered for character in "✅❌⚠️⏳")


def test_mixie_validates_result_dispatch_and_bundle():
    run_id, run_attempt = validate_dispatch(
        {
            "action": "ci-run-result",
            "client_payload": {
                "schema_version": 1,
                "run_id": 1842,
                "run_attempt": 2,
            },
        }
    )
    attempt = {
        "repository": "Marvis-Labs/mlx-vlm",
        "pull_request": 42,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
    }
    job = {
        "id": "model-path-qwen2_vl",
        "repository": attempt["repository"],
        "base_sha": attempt["base_sha"],
        "head_sha": attempt["head_sha"],
        "manifest_digest": "c" * 64,
    }
    result = {"job_id": job["id"], "manifest_digest": job["manifest_digest"]}
    bundle = {
        "schema_version": 1,
        "attempt": attempt,
        "jobs": {"jobs": [job]},
        "results": [result],
        "run_url": "https://github.com/Marvis-Labs/mlx-ci/actions/runs/1842",
    }
    validate_bundle(bundle, attempt["repository"], run_id, run_attempt)
    bundle["results"][0]["manifest_digest"] = "d" * 64
    try:
        validate_bundle(bundle, attempt["repository"], run_id, run_attempt)
    except OutputError:
        pass
    else:
        raise AssertionError("unbound result was accepted")


def test_mixie_renders_missing_device_as_unavailable():
    attempt = {
        "run_id": 1842,
        "run_attempt": 1,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "changed_files": ["mlx_vlm/server/openai.py"],
    }
    job = plan_ci(attempt["changed_files"], CATALOG)["jobs"][0]
    job.update(manifest_digest="c" * 64)
    result = {
        "job_id": job["id"],
        "manifest_digest": job["manifest_digest"],
        "status": "infrastructure_failure",
        "device": None,
        "cache": "not_applicable",
        "duration_ms": 0,
        "checks": [
            {
                "name": "Runner",
                "category": "infrastructure",
                "status": "infrastructure_failure",
                "detail": "No eligible runner reported a result",
            }
        ],
        "metrics": [],
    }
    rendered = render_comment(
        attempt,
        {"jobs": [job], "blocked": []},
        [result],
        "https://github.com/Marvis-Labs/mlx-ci/actions/runs/1842",
    )
    assert "Device: unavailable" in rendered


def test_mixie_renders_blocked_work_as_a_terminal_section():
    attempt = {
        "run_id": 1842,
        "run_attempt": 1,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "changed_files": ["mlx_vlm/models/new_family/model.py"],
    }
    rendered = render_comment(
        attempt,
        plan_ci(attempt["changed_files"], CATALOG),
        [],
        "https://github.com/Marvis-Labs/mlx-ci/actions/runs/1842",
    )
    assert "Blocked — 0 of 1 sections passed" in rendered
    assert "<strong>new_family</strong> · ModelPath · Blocked" in rendered
    assert "No synthetic model case is registered for this family." in rendered


def test_mixie_renders_each_coalesced_command_as_a_new_notice():
    pull_request, rendered = render_coalesced(
        {
            "action": "ci-run-coalesced",
            "client_payload": {
                "schema_version": 1,
                "pull_request": 42,
                "comment_id": 987,
                "requested_at": "2026-09-28T14:30:00Z",
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
            },
        }
    )
    assert pull_request == 42
    assert "mixie:coalesced:987" in rendered
    assert "No duplicate runner work was started" in rendered


def test_runner_execution_rejects_manifest_tampering():
    job = {
        "schema_version": 2,
        "engine": "vlm",
        "repository": "Marvis-Labs/mlx-vlm",
        "pull_request": 42,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "head_repository": "contributor/mlx-vlm",
        "contract_sha": "a" * 40,
        "id": "model-path-qwen2_vl",
        "component": "model_path",
        "subject": "qwen2_vl",
        "phases": ["synthetic"],
        "work": {"synthetic": {"selectors": ["model-contract"]}},
        "resources": {},
        "artifact": None,
        "estimated_peak_bytes": 1,
        "required_memory_gib": 16,
        "required_disk_gib": 4,
    }
    encoded = json.dumps(
        job, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    job["manifest_digest"] = hashlib.sha256(encoded).hexdigest()
    validate_job(job)
    job["head_sha"] = "c" * 40
    try:
        validate_job(job)
    except ExecutionSecurityError:
        pass
    else:
        raise AssertionError("tampered manifest was accepted")
