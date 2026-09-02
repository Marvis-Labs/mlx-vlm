import ast
import json
import subprocess
from pathlib import Path

import pytest

from ci.kv_cache_contract_compare import (
    TRUSTED_HARNESS_FILES,
    execute,
    require_checkout,
    run_probe,
)


def test_batch_contract_adapter_is_part_of_the_trusted_harness():
    assert "ci/kv_cache_batch.py" in TRUSTED_HARNESS_FILES


def test_profile_contracts_do_not_import_one_another():
    profiles = Path("ci/kv_cache_profiles")
    violations = []
    for path in profiles.glob("*.py"):
        if path.name in {"__init__.py", "common.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("ci.kv_cache_profiles.")
                and node.module != "ci.kv_cache_profiles.common"
            ):
                violations.append((path.name, node.module))

    assert violations == []


def job():
    return {
        "head_sha": "head",
        "contract_sha": "contract",
        "kv_cache_contract": {
            "profile": "dense",
            "entry_point": "ci.kv_cache_profiles.dense:dense_contract_cases",
        },
    }


def test_execute_validates_both_revisions_and_reports_oracle_identity(
    tmp_path, monkeypatch
):
    calls = []

    def check(path, expected, role):
        calls.append((path, expected, role))
        return expected

    monkeypatch.setattr("ci.kv_cache_contract_compare.require_checkout", check)
    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.run_probe",
        lambda head, control, probe, profiles, output: {
            "component": "kv_cache_change",
            "verdict": "passed",
            "checks": 1668,
            "cases": [],
            "implementation_path": str((tmp_path / "head/mlx_vlm/models/cache.py")),
        },
    )

    result = execute(
        job(),
        tmp_path / "control",
        tmp_path / "head",
        tmp_path / "control/ci/probe.py",
        tmp_path / "result.json",
    )

    assert calls == [
        (tmp_path / "control", "contract", "contract"),
        (tmp_path / "head", "head", "head"),
    ]
    assert result["head_sha"] == "head"
    assert result["contract_sha"] == "contract"
    assert result["correctness"] == {
        "match": True,
        "oracle": "trusted_independent_semantic_contract",
    }


def test_checkout_identity_rejects_a_different_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.checkout_sha", lambda path: "other"
    )

    with pytest.raises(RuntimeError, match="expected expected"):
        require_checkout(tmp_path, "expected", "head")


def test_probe_must_reside_in_trusted_control_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="trusted control checkout"):
        run_probe(
            tmp_path / "head",
            tmp_path / "control",
            tmp_path / "outside/probe.py",
            ("dense",),
            tmp_path / "result.json",
        )


def test_probe_command_uses_head_project_and_trusted_python_path(tmp_path, monkeypatch):
    control = tmp_path / "control"
    probe = control / "ci/kv_cache_contract_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("")
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"verdict": "passed"}))
    observed = {}

    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.require_tracked_file",
        lambda repository, path: None,
    )
    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.shutil.copy2", lambda source, destination: None
    )

    def invoke(command, *, env):
        observed["command"] = command
        observed["environment"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ci.kv_cache_contract_compare.subprocess.run", invoke)

    result = run_probe(
        tmp_path / "head",
        control,
        probe,
        ("ci.kv_cache_profiles.dense:dense_contract_cases",),
        output,
    )

    assert result["verdict"] == "passed"
    assert observed["command"][0:7] == [
        "uv",
        "run",
        "--frozen",
        "--offline",
        "--project",
        str(tmp_path / "head"),
        "--python",
    ]
    assert observed["environment"]["PYTHONPATH"] != str(control)
    assert "mlx-vlm-ci-contract-" in observed["environment"]["PYTHONPATH"]


def test_probe_preserves_structured_contract_failures(tmp_path, monkeypatch):
    control = tmp_path / "control"
    probe = control / "ci/kv_cache_contract_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("")
    output = tmp_path / "result.json"
    failure = {
        "verdict": "test_failure",
        "cases": [
            {
                "case": "KVCache",
                "failures": [
                    {
                        "sequence": "append-trim-resume",
                        "step": 2,
                        "characteristic": "content",
                    }
                ],
            }
        ],
    }

    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.require_tracked_file",
        lambda repository, path: None,
    )
    monkeypatch.setattr(
        "ci.kv_cache_contract_compare.shutil.copy2", lambda source, destination: None
    )

    def invoke(command, *, env):
        output.write_text(json.dumps(failure))
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr("ci.kv_cache_contract_compare.subprocess.run", invoke)

    result = run_probe(
        tmp_path / "head",
        control,
        probe,
        ("ci.kv_cache_profiles.dense:dense_contract_cases",),
        output,
    )

    assert result == failure
