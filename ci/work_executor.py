from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.execution_security import ExecutionSecurityError, verify_execution

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
import tempfile
import time
import wave
from pathlib import Path

from PIL import Image

sys.path.insert(0, sys.argv[1])
from mlx_vlm import apply_chat_template, generate, load

path = os.environ["CI_CHECKPOINT_PATH"]
configuration = json.loads(sys.argv[3])
assets = Path(os.environ.get("CI_ASSETS_ROOT", str(Path(sys.argv[4]) / "ci_assets")))

def image_fixture(relative):
    value = json.loads((assets / relative).read_text())
    image = Image.new(value["mode"], (value["width"], value["height"]))
    image.putdata([tuple(pixel) for pixel in value["pixels"]])
    target = Path(tempfile.mkdtemp()) / "image.png"
    image.save(target)
    return str(target)

def audio_fixture(relative):
    value = json.loads((assets / relative).read_text())
    target = Path(tempfile.mkdtemp()) / "audio.wav"
    samples = value["samples"] * value["repeat"]
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(value["sample_rate"])
        stream.writeframes(b"".join(int(max(-1, min(1, sample)) * 32767).to_bytes(2, "little", signed=True) for sample in samples))
    return str(target)

profile = configuration["profile"]
image = audio = None
if profile == "image":
    image = [image_fixture(configuration["asset"])]
elif profile == "audio":
    audio = [audio_fixture(configuration["asset"])]
elif profile == "omni":
    image = [image_fixture(configuration["assets"][0])]
    audio = [audio_fixture(configuration["assets"][1])]
else:
    raise ValueError("unsupported checkpoint profile")

model, processor = load(path)
prompt = apply_chat_template(
    processor,
    model.config,
    configuration["prompt"],
    num_images=len(image or []),
    num_audios=len(audio or []),
)
started = time.perf_counter()
result = generate(
    model,
    processor,
    prompt,
    image=image,
    audio=audio,
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
    return {
        "name": "Synthetic structure",
        "category": "correctness",
        "status": "passed",
        "detail": "Tiny random-weight contracts passed on main and PR",
    }


def _checkpoint(
    job: Mapping[str, Any], control: Path, base: Path, head: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configuration = job["work"]["checkpoint"]
    max_tokens = configuration.get("max_tokens")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("checkpoint token count is invalid")
    encoded = json.dumps(configuration, separators=(",", ":"))
    observations = []
    for project in (base, head):
        result = _run(
            [
                sys.executable,
                "-c",
                CHECKPOINT_PROBE,
                str(project),
                str(max_tokens),
                encoded,
                str(control),
            ],
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
    return {
        "name": "Checkpoint output",
        "category": "correctness",
        "status": "passed" if match else "failed",
        "detail": (
            "Main and PR generated the same text"
            if match
            else "Main and PR generated different text"
        ),
    }, metrics


def _test_summary(result: subprocess.CompletedProcess[str]) -> str:
    stdout = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    stderr = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    summary = stdout[-1] if stdout else stderr[-1] if stderr else "no output"
    return " ".join(summary.split())[:70]


def _server_contract(job: Mapping[str, Any], base: Path, head: Path) -> dict[str, Any]:
    configuration = job["work"]["server_contract"]
    profiles = configuration.get("profiles")
    selectors = configuration.get("selectors")
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(profile, str) or not profile for profile in profiles)
        or not isinstance(selectors, list)
        or selectors != ["mlx_vlm/tests/test_server.py"]
    ):
        raise ValueError("server contract configuration is invalid")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "-p",
        "no:cacheprovider",
        *selectors,
    ]
    base_result = _run(command, base)
    head_result = _run(command, head)
    passed = head_result.returncode == 0
    profile_names = ", ".join(profiles)
    detail = (
        f"Main: {_test_summary(base_result)}; PR: {_test_summary(head_result)}; "
        f"profiles: {profile_names}"
    )[:160]
    return {
        "name": "Server contract",
        "category": "correctness",
        "status": "passed" if passed else "failed",
        "detail": detail,
    }


def execute(
    job: Mapping[str, Any], control: Path, base: Path, head: Path
) -> dict[str, Any]:
    checks, metrics = [], []
    for phase in job["phases"]:
        if phase == "synthetic":
            checks.append(_synthetic(job, base, head))
        elif phase == "checkpoint":
            check, metrics = _checkpoint(job, control, base, head)
            checks.append(check)
        elif phase == "server_contract":
            checks.append(_server_contract(job, base, head))
    matched = all(check["status"] == "passed" for check in checks)
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
        findings = execute(job, arguments.control, arguments.base, arguments.head)
        findings["duration_ms"] = int((time.perf_counter() - started) * 1000)
    except Exception as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            message = error.stderr.strip()
        findings = {
            "verdict": (
                "infrastructure_failure"
                if isinstance(error, ExecutionSecurityError)
                else "test_failure"
            ),
            "error": f"{type(error).__name__}: {message}"[-500:],
            "metrics": [],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
    output.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    return 0 if findings["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
