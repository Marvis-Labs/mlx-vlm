from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.execution_security import verify_execution

MODEL_PROBE = r"""
import importlib
import json
import sys
import types

class Mark:
    def parametrize(self, *args, **kwargs):
        return lambda function: function

pytest = types.ModuleType("pytest")
pytest.mark = Mark()
pytest.raises = lambda *args, **kwargs: None
sys.modules["pytest"] = pytest
project, selectors = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, project)
tests = importlib.import_module("mlx_vlm.tests.test_models")
cases = {case["id"]: case for case in tests.DATA["cases"]}
for selector in selectors:
    name = selector.rsplit("::", 1)[-1]
    if name.startswith("test_model_contract[") and name.endswith("]"):
        tests.test_model_contract(cases[name[20:-1]])
    elif name.startswith("test_dense_model[") and name.endswith("]"):
        tests.test_dense_model(name[17:-1])
    else:
        raise ValueError(f"unsupported synthetic selector: {selector}")
"""

CHECKPOINT_PROBE = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
from mlx_vlm import apply_chat_template, generate, load

path = os.environ["CI_CHECKPOINT_PATH"]
model, processor = load(path)
prompt = apply_chat_template(
    processor,
    model.config,
    "Reply with exactly one short word.",
    num_images=0,
)
started = time.perf_counter()
result = generate(
    model,
    processor,
    prompt,
    image=None,
    max_tokens=int(sys.argv[2]),
    temperature=0.0,
    verbose=False,
)
wall_ms = (time.perf_counter() - started) * 1000
ttft_ms = (
    result.prompt_tokens / result.prompt_tps * 1000 if result.prompt_tps > 0 else wall_ms
)
print(json.dumps({
    "text": result.text,
    "prefill_tps": result.prompt_tps,
    "decode_tps": result.generation_tps,
    "ttft_ms": ttft_ms,
    "wall_ms": wall_ms,
    "peak_memory_gib": result.peak_memory,
}, sort_keys=True))
"""


def _run(command: list[str], project: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project)
    return subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
    )


def _synthetic(job: Mapping[str, Any], base: Path, head: Path) -> dict[str, Any]:
    selectors = job["work"]["synthetic"].get("selectors")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("synthetic selectors are invalid")
    encoded = json.dumps(selectors, separators=(",", ":"))
    for project in (base, head):
        result = _run(
            [sys.executable, "-c", MODEL_PROBE, str(project), encoded], project
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-500:] or "synthetic probe failed")
    return {"name": "Synthetic", "match": True}


def _checkpoint(
    job: Mapping[str, Any], base: Path, head: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configuration = job["work"]["checkpoint"]
    max_tokens = configuration.get("max_tokens")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("checkpoint token count is invalid")
    observations = []
    for project in (base, head):
        result = _run(
            [sys.executable, "-c", CHECKPOINT_PROBE, str(project), str(max_tokens)],
            project,
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip()[-500:] or "checkpoint probe failed"
            )
        observations.append(json.loads(result.stdout.strip().splitlines()[-1]))
    base_result, head_result = observations
    metrics = []
    for name, unit, higher_is_better in (
        ("prefill_tps", "tok/s", True),
        ("decode_tps", "tok/s", True),
        ("ttft_ms", "ms", False),
        ("wall_ms", "ms", False),
        ("peak_memory_gib", "GiB", False),
    ):
        base_value, head_value = float(base_result[name]), float(head_result[name])
        change = (
            0.0 if base_value == 0 else (head_value - base_value) / base_value * 100
        )
        improved = change >= 4 if higher_is_better else change <= -4
        regressed = change <= -4 if higher_is_better else change >= 4
        verdict = "improved" if improved else "regressed" if regressed else "stable"
        metrics.append(
            {
                "name": name,
                "unit": unit,
                "base": base_value,
                "head": head_value,
                "change_pct": change,
                "verdict": verdict,
            }
        )
    match = base_result["text"] == head_result["text"]
    if not match:
        for metric in metrics:
            metric["verdict"] = "advisory"
    return {"name": "Checkpoint", "match": match}, metrics


def execute(job: Mapping[str, Any], base: Path, head: Path) -> dict[str, Any]:
    checks, metrics = [], []
    for phase in job["phases"]:
        if phase == "synthetic":
            checks.append(_synthetic(job, base, head))
        elif phase == "checkpoint":
            check, metrics = _checkpoint(job, base, head)
            checks.append(check)
        elif phase == "server_contract":
            raise RuntimeError("server contract execution is not registered")
    matched = all(check["match"] for check in checks)
    return {
        "verdict": "passed" if matched else "regressed",
        "checks": checks,
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    arguments = parser.parse_args(argv)
    output = Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    started = time.perf_counter()
    try:
        job = json.loads(arguments.job.read_text(encoding="utf-8"))
        verify_execution(
            job,
            control=arguments.control,
            base=arguments.base,
            head=arguments.head,
        )
        findings = execute(job, arguments.base, arguments.head)
        findings["duration_ms"] = int((time.perf_counter() - started) * 1000)
    except Exception as error:
        findings = {
            "verdict": "test_failure",
            "error": f"{type(error).__name__}: {error}"[:500],
            "metrics": [],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
    output.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    return 0 if findings["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
