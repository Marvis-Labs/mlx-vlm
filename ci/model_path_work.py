from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _run(command: list[str], findings: Path) -> tuple[int, dict[str, Any]]:
    environment = dict(os.environ)
    environment["CI_JOB_FINDINGS"] = str(findings)
    completed = subprocess.run(command, env=environment)
    if not findings.is_file():
        return completed.returncode or 1, {
            "verdict": "test_failure",
            "error": "phase produced no findings",
        }
    value = json.loads(findings.read_text())
    if not isinstance(value, Mapping):
        return completed.returncode or 1, {
            "verdict": "test_failure",
            "error": "phase findings must be an object",
        }
    return completed.returncode, dict(value)


def _phase(findings: Mapping[str, Any], returncode: int) -> dict[str, Any]:
    verdict = str(findings.get("verdict", "test_failure"))
    outcome = (
        verdict if verdict in {"passed", "improved", "regressed"} else "test_failure"
    )
    if returncode and outcome != "test_failure":
        outcome = "test_failure"
    return {"outcome": outcome, "findings": dict(findings)}


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    synthetic_path = output.with_name(output.stem + "-synthetic.json")
    checkpoint_path = output.with_name(output.stem + "-hf.json")
    synthetic_code, synthetic_findings = _run(
        [
            sys.executable,
            str(args.synthetic_compare),
            "--job",
            str(args.job),
            "--profiles",
            str(args.profiles),
            "--base",
            str(args.base),
            "--head",
            str(args.head),
            "--probe",
            str(args.synthetic_probe),
        ],
        synthetic_path,
    )
    phases: dict[str, Any] = {"synthetic": _phase(synthetic_findings, synthetic_code)}
    if phases["synthetic"]["outcome"] == "test_failure":
        phases["hf_checkpoint"] = {
            "outcome": "skipped",
            "findings": {"reason": "synthetic_failed"},
        }
        result = {"verdict": "test_failure", "phases": phases}
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 2, result

    checkpoint_code, checkpoint_findings = _run(
        [
            sys.executable,
            str(args.hf_compare),
            "--job",
            str(args.job),
            "--base",
            str(args.base),
            "--head",
            str(args.head),
            "--probe",
            str(args.hf_probe),
            "--image",
            str(args.image),
            "--max-tokens",
            str(args.max_tokens),
        ],
        checkpoint_path,
    )
    phases["hf_checkpoint"] = _phase(checkpoint_findings, checkpoint_code)
    outcomes = {phase["outcome"] for phase in phases.values()}
    if "test_failure" in outcomes:
        verdict = "test_failure"
    elif "regressed" in outcomes:
        verdict = "regressed"
    elif "improved" in outcomes:
        verdict = "improved"
    else:
        verdict = "passed"
    result = {"verdict": verdict, "phases": phases}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return (2 if verdict == "test_failure" else 0), result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--synthetic-compare", type=Path, required=True)
    parser.add_argument("--synthetic-probe", type=Path, required=True)
    parser.add_argument("--hf-compare", type=Path, required=True)
    parser.add_argument("--hf-probe", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    return run(args)[0]


if __name__ == "__main__":
    raise SystemExit(main())
