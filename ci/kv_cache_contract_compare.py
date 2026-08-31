from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

TRUSTED_HARNESS_FILES = (
    "ci/__init__.py",
    "ci/kv_cache_contract.py",
    "ci/kv_cache_oracles.py",
    "ci/kv_cache_contract_probe.py",
    "ci/kv_cache_profiles/__init__.py",
    "ci/kv_cache_profiles/dense.py",
)


def checkout_sha(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_checkout(repository: Path, expected_sha: str, role: str) -> str:
    actual = checkout_sha(repository)
    if actual != expected_sha:
        raise RuntimeError(f"{role} checkout is {actual}, expected {expected_sha}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(f"{role} checkout contains tracked modifications")
    return actual


def require_tracked_file(repository: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise RuntimeError("trusted probe is outside the contract checkout") from error
    expected = subprocess.run(
        ["git", "rev-parse", f"HEAD:{relative.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    actual = subprocess.run(
        ["git", "hash-object", str(path.resolve())],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError("trusted probe content does not match the contract commit")


def run_probe(
    head: Path,
    control: Path,
    probe: Path,
    profiles: Sequence[str],
    output: Path,
) -> Mapping[str, Any]:
    trusted_ci = (control / "ci").resolve()
    resolved_probe = probe.resolve()
    expected_probe = trusted_ci / "kv_cache_contract_probe.py"
    if resolved_probe != expected_probe:
        raise RuntimeError("KV cache probe must come from the trusted control checkout")
    for relative in TRUSTED_HARNESS_FILES:
        require_tracked_file(control, control / relative)
    with tempfile.TemporaryDirectory(prefix="mlx-vlm-ci-contract-") as directory:
        harness = Path(directory)
        for relative in TRUSTED_HARNESS_FILES:
            destination = harness / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(control / relative, destination)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(harness)
        command = [
            "uv",
            "run",
            "--frozen",
            "--project",
            str(head),
            "--python",
            "3.10",
            "python",
            str(harness / "ci/kv_cache_contract_probe.py"),
            "--output",
            str(output),
        ]
        for profile in profiles:
            command.extend(("--profile", profile))
        completed = subprocess.run(command, env=environment)
        if completed.returncode not in {0, 2}:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if not output.is_file():
            raise RuntimeError("KV cache contract probe produced no findings")
    value = json.loads(output.read_text())
    if not isinstance(value, Mapping):
        raise RuntimeError("KV cache contract probe output must be an object")
    return value


def execute(
    job: Mapping[str, Any], control: Path, head: Path, probe: Path, output: Path
) -> dict[str, Any]:
    expected_head = job.get("head_sha")
    expected_contract = job.get("contract_sha")
    contract = job.get("kv_cache_contract")
    if not isinstance(expected_head, str) or not expected_head:
        raise ValueError("KVCacheChange work has no immutable head_sha")
    if not isinstance(expected_contract, str) or not expected_contract:
        raise ValueError("KVCacheChange work has no immutable contract_sha")
    if not isinstance(contract, Mapping):
        raise ValueError("KVCacheChange work has no contract configuration")
    profile = contract.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("KVCacheChange contract has no profile")
    contract_sha = require_checkout(control, expected_contract, "contract")
    head_sha = require_checkout(head, expected_head, "head")
    result = dict(run_probe(head, control, probe, (profile,), output))
    implementation_path = result.get("implementation_path")
    if not isinstance(implementation_path, str) or not Path(
        implementation_path
    ).resolve().is_relative_to(head.resolve()):
        raise RuntimeError("KV cache implementation was not imported from PR head")
    match = result.get("verdict") == "passed"
    result.update(
        {
            "head_sha": head_sha,
            "contract_sha": contract_sha,
            "correctness": {
                "match": match,
                "oracle": "trusted_independent_semantic_contract",
            },
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args(argv)
    findings = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    try:
        job = json.loads(args.job.read_text())
        result = execute(job, args.control, args.head, args.probe, findings)
    except Exception as error:
        result = {
            "verdict": "test_failure",
            "correctness": {"match": False},
            "error": f"{type(error).__name__}: {error}",
        }
    findings.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
