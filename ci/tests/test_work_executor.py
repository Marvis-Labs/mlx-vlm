import argparse
import json
from types import SimpleNamespace

from ci.model_path_synthetic_compare import compare
from ci.work_executor import run


def work_args(tmp_path, phases):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"phases": phases}))
    return argparse.Namespace(
        job=job,
        base=tmp_path / "base",
        head=tmp_path / "head",
        control=tmp_path / "control",
        image=None,
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

    monkeypatch.setattr("ci.work_executor._run", fake_run)
    output = tmp_path / "findings.json"
    monkeypatch.setenv("CI_JOB_FINDINGS", str(output))
    args = work_args(tmp_path, ["synthetic", "hf_checkpoint"])

    code, result = run(args, validate_execution=False)

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

    monkeypatch.setattr("ci.work_executor._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))
    args = work_args(tmp_path, ["synthetic", "hf_checkpoint"])

    code, result = run(args, validate_execution=False)

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

    monkeypatch.setattr("ci.work_executor._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))

    code, result = run(
        work_args(tmp_path, ["mlp_contract", "hf_checkpoint"]),
        validate_execution=False,
    )

    assert code == 0
    assert list(result["phases"]) == ["mlp_contract", "hf_checkpoint"]
    assert calls[0][1].endswith("ci/mlp_contract_compare.py")
    assert calls[1][1].endswith("ci/model_path_compare.py")


def test_kv_cache_contract_runs_only_against_head(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, path):
        calls.append(command)
        return 0, {"verdict": "passed", "correctness": {"match": True}}

    monkeypatch.setattr("ci.work_executor._run", fake_run)
    monkeypatch.setenv("CI_JOB_FINDINGS", str(tmp_path / "findings.json"))

    code, result = run(
        work_args(tmp_path, ["kv_cache_contract"]), validate_execution=False
    )

    assert code == 0
    assert result["verdict"] == "passed"
    command = calls[0]
    assert command[1].endswith("ci/kv_cache_contract_compare.py")
    assert "--head" in command
    assert "--control" in command
    assert "--base" not in command


def test_phase_environment_does_not_forward_runner_secrets(monkeypatch, tmp_path):
    findings = tmp_path / "phase.json"
    captured = {}

    def fake_subprocess(command, env):
        captured.update(env)
        findings.write_text('{"verdict":"passed"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ci.work_executor.subprocess.run", fake_subprocess)
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("RUNNER_TOKEN", "secret")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from ci.work_executor import _run

    code, _ = _run(["probe"], findings)

    assert code == 0
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in captured
    assert "GH_TOKEN" not in captured
    assert "RUNNER_TOKEN" not in captured
