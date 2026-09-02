import argparse

from ci import repository_adapter


def test_prepare_record_emits_central_device_job_contract(monkeypatch):
    planned = {
        "schema_version": 1,
        "kind": "ci_control",
        "repository": "Marvis-Labs/mlx-vlm",
        "pr_number": 1,
        "base_sha": "a" * 40,
        "target_sha": "a" * 40,
        "head_sha": "b" * 40,
        "contract_sha": "c" * 40,
        "outcome": "ready",
        "run_url": "https://example.invalid/run",
        "rules": [],
        "components": [],
        "jobs": [
            {
                "id": "model_path:example",
                "work_type": "ModelPath",
                "component": "model_path",
                "subject": "example",
                "model": "example",
                "phases": ["synthetic", "hf_checkpoint"],
                "required_memory_gib": 8,
                "required_disk_gib": 4,
                "synthetic": {"adapter": "example", "profile": "dense_vlm"},
                "hf_checkpoint": {
                    "repo": "example/model",
                    "revision": "d" * 40,
                    "expected_model_type": "example",
                    "weight": {"bytes": 1024},
                },
                "scenarios": ["vlm_animal"],
            }
        ],
        "gates": [],
        "checks": [],
        "errors": [],
    }
    monkeypatch.setattr(repository_adapter, "build_plan", lambda args: planned)
    args = argparse.Namespace(
        attempt_id="attempt-1",
        repository="Marvis-Labs/mlx-vlm",
        base_sha="a" * 40,
        head_sha="b" * 40,
        contract_sha="c" * 40,
    )

    record = repository_adapter.prepare_record(args)

    assert record["terminal_state"] == "planned"
    assert record["device_jobs"][0]["id"] == "model_path:example"
    manifest = record["device_jobs"][0]["manifest"]
    assert manifest["repository"] == "Marvis-Labs/mlx-vlm"
    assert manifest["manifest_digest"].startswith("sha256:")


def test_invalid_repository_configuration_becomes_a_blocked_plan(monkeypatch):
    def fail_materialize(*args, **kwargs):
        raise ValueError("invalid yaml")

    monkeypatch.setattr(repository_adapter, "materialize", fail_materialize)
    args = argparse.Namespace(
        repository_path=None,
        base_checkout=None,
        head_checkout=None,
        base_sha="a" * 40,
        head_sha="b" * 40,
        contract_sha="c" * 40,
        repository="Marvis-Labs/mlx-vlm",
        pr_number=1,
        run_url="https://example.invalid/run",
    )

    record = repository_adapter.build_plan(args)

    assert record["outcome"] == "blocked"
    assert record["errors"][0]["code"] == "invalid_ci_configuration"
