import argparse
import json

from ci.model_path_synthetic_compare import compare
from ci.model_path_work import run


def work_args(tmp_path, phases):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"phases": phases}))
    return argparse.Namespace(
        synthetic_compare=tmp_path / "synthetic-compare.py",
        synthetic_probe=tmp_path / "synthetic-probe.py",
        mlp_compare=tmp_path / "mlp-compare.py",
        mlp_probe=tmp_path / "mlp-probe.py",
        kv_cache_compare=tmp_path / "kv-cache-compare.py",
        kv_cache_probe=tmp_path / "kv-cache-probe.py",
        hf_compare=tmp_path / "hf-compare.py",
        hf_probe=tmp_path / "hf-probe.py",
        job=job,
        profiles=tmp_path / "profiles.yaml",
        base=tmp_path / "base",
        head=tmp_path / "head",
        control=tmp_path / "control",
        image=tmp_path / "cat.jpg",
        max_tokens=16,
    )


def test_synthetic_compare_passes_without_numpy_in_control_process():
    base = {
        "output": [1.0, 2.0],
        "output_shape": [1, 2],
        "parameter_signature": "same",
        "finite": True,
        "output_hash": "base",
    }
    head = {**base, "output": [1.0, 2.000001], "output_hash": "head"}

    result = compare(base, head)

    assert result["verdict"] == "passed"
    assert result["correctness"]["match"] is True


def test_synthetic_failure_skips_hf_checkpoint(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, findings):
        calls.append(command)
        return 2, {"verdict": "test_failure", "error": "shape mismatch"}

    monkeypatch.setattr("ci.model_path_work._run", fake_run)
    output = tmp_path / "findings.json"
    monkeypatch.setenv("CI_JOB_FINDINGS", str(output))
    args = work_args(tmp_path, ["synthetic", "hf_checkpoint"])

    code, result = run(args)

    assert code == 2
    assert len(calls) == 1
    assert result["phases"]["synthetic"]["outcome"] == "test_failure"
    assert result["phases"]["hf_checkpoint"]["outcome"] == "skipped"
    assert json.loads(output.read_text()) == result


def test_hf_checkpoint_runs_only_after_synthetic_passes(monkeypatch, tmp_path):
    findings = iter(
        [
            (0, {"verdict": "passed", "correctness": {"match": True}}),
            (0, {"verdict": "improved", "correctness": {"match": True}}),
        ]
    )
    calls = []

    def fake_run(command, path):
        calls.append(command)
        return next(findings)

    monkeypatch.setattr("ci.model_path_work._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))
    args = work_args(tmp_path, ["synthetic", "hf_checkpoint"])

    code, result = run(args)

    assert code == 0
    assert len(calls) == 2
    assert result["verdict"] == "improved"


def test_mlp_contract_runs_before_optional_checkpoint(monkeypatch, tmp_path):
    findings = iter(
        [
            (0, {"verdict": "passed", "correctness": {"match": True}}),
            (0, {"verdict": "passed", "correctness": {"match": True}}),
        ]
    )
    calls = []

    def fake_run(command, path):
        calls.append(command)
        return next(findings)

    monkeypatch.setattr("ci.model_path_work._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))

    code, result = run(work_args(tmp_path, ["mlp_contract", "hf_checkpoint"]))

    assert code == 0
    assert list(result["phases"]) == ["mlp_contract", "hf_checkpoint"]
    assert "mlp-compare.py" in calls[0][1]
    assert "hf-compare.py" in calls[1][1]


def test_kv_cache_contract_runs_only_against_head(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, path):
        calls.append(command)
        return 0, {"verdict": "passed", "correctness": {"match": True}}

    monkeypatch.setattr("ci.model_path_work._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))

    code, result = run(work_args(tmp_path, ["kv_cache_contract"]))

    assert code == 0
    assert result["verdict"] == "passed"
    command = calls[0]
    assert "kv-cache-compare.py" in command[1]
    assert "--head" in command
    assert "--control" in command
    assert "--base" not in command
