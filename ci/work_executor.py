from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.execution_security import ExecutionSecurityError, verify_execution

CHECKPOINT_PROBE = r"""
import hashlib
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, sys.argv[1])
from mlx_vlm import apply_chat_template, batch_generate, generate, load
from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.models.pooling import read_pooling_config
from mlx_vlm.reranker_loader import load_reranker
from mlx_vlm.server.reranking import RerankItem, score_documents
from mlx_vlm.utils import load_config, load_processor

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

def generation_probe():
    profile = configuration["profile"]
    image = audio = None
    if profile == "image":
        image = image_fixture(configuration["asset"])
    elif profile == "audio":
        audio = audio_fixture(configuration["asset"])
    elif profile == "omni":
        image = image_fixture(configuration["assets"][0])
        audio = audio_fixture(configuration["assets"][1])
    elif profile == "text":
        pass
    else:
        raise ValueError("unsupported generation profile")
    model, processor = load(path)
    batch_size = configuration.get("batch_size", 1)
    prompt = apply_chat_template(
        processor,
        model.config,
        configuration["prompt"],
        num_images=int(image is not None),
        num_audios=int(audio is not None),
    )
    started = time.perf_counter()
    if batch_size == 1:
        result = generate(
            model,
            processor,
            prompt,
            image=[image] if image else None,
            audio=[audio] if audio else None,
            max_tokens=int(sys.argv[2]),
            temperature=0.0,
            verbose=False,
        )
        texts = [result.text]
        stats = result
    else:
        result = batch_generate(
            model,
            processor,
            images=[image] * batch_size if image else None,
            audios=[audio] * batch_size if audio else None,
            prompts=[prompt] * batch_size,
            max_tokens=int(sys.argv[2]),
            verbose=False,
        )
        texts = result.texts
        stats = result.stats
    wall_ms = (time.perf_counter() - started) * 1000
    ttft_ms = stats.prompt_tokens / stats.prompt_tps * 1000 if stats.prompt_tps > 0 else wall_ms
    return {
        "signature": texts,
        "metrics": {
            "prefill_tps": stats.prompt_tps,
            "decode_tps": stats.generation_tps,
            "ttft_ms": ttft_ms,
            "wall_ms": wall_ms,
            "peak_memory_gib": stats.peak_memory,
        },
    }

def embedding_probe():
    model_dir = Path(path)
    model = load_embedding_model(model_dir)
    model.pooling_config = read_pooling_config(model_dir)
    processor = load_processor(model_dir, add_detokenizer=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    texts = configuration["texts"]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=configuration["prompt_tokens"],
        return_tensors="np",
    )
    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = model(input_ids, attention_mask=attention_mask).text_embeds
    mx.eval(output)
    wall_ms = (time.perf_counter() - started) * 1000
    vectors = np.asarray(output.astype(mx.float32))
    if "sentence_embeddings" in configuration["model_checks"]:
        scores = vectors[0] @ vectors[1:].T
    else:
        mask = np.asarray(encoded["attention_mask"], dtype=bool)
        query = vectors[0][mask[0]]
        scores = np.asarray([
            np.max(query @ vectors[index][mask[index]].T, axis=1).sum()
            for index in range(1, len(vectors))
        ])
    if float(scores[0]) <= float(scores[1]):
        raise RuntimeError("checkpoint failed the semantic ordering fixture")
    return {
        "signature": {
            "digest": hashlib.sha256(vectors.tobytes()).hexdigest(),
            "shape": list(vectors.shape),
            "positive_first": True,
        },
        "metrics": {
            "items_per_second": len(texts) / max(wall_ms / 1000, 1e-9),
            "wall_ms": wall_ms,
            "peak_memory_gib": mx.get_peak_memory() / 1e9,
        },
    }

def rerank_probe():
    model, processor = load_reranker(path)
    config = load_config(Path(path))
    query = RerankItem(text=configuration["query"])
    documents = [RerankItem(text=text) for text in configuration["documents"]]
    mx.reset_peak_memory()
    started = time.perf_counter()
    scores, prompt_tokens = score_documents(
        model,
        processor,
        config,
        query,
        documents,
        configuration["instruction"],
    )
    mx.eval(scores)
    wall_ms = (time.perf_counter() - started) * 1000
    if scores[0] <= scores[1]:
        raise RuntimeError("checkpoint failed the reranking fixture")
    return {
        "signature": {
            "scores": [round(float(score), 6) for score in scores],
            "positive_first": True,
        },
        "metrics": {
            "items_per_second": len(documents) / max(wall_ms / 1000, 1e-9),
            "wall_ms": wall_ms,
            "peak_memory_gib": mx.get_peak_memory() / 1e9,
            "prompt_tokens": prompt_tokens,
        },
    }

profile = configuration["profile"]
if profile == "embedding":
    payload = embedding_probe()
elif profile == "rerank":
    payload = rerank_probe()
else:
    payload = generation_probe()
print(json.dumps(payload, sort_keys=True))
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


def _synthetic(
    job: Mapping[str, Any], control: Path, base: Path, head: Path
) -> dict[str, Any]:
    selectors = job["work"]["synthetic"].get("selectors")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("synthetic selectors are invalid")
    tests = []
    for selector in selectors:
        path, separator, node = selector.partition("::")
        if not separator or not node:
            raise ValueError("synthetic selector is invalid")
        tests.append(f"{control / path}::{node}")
    for label, project in (("Main", base), ("PR", head)):
        result = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--disable-warnings",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                "--rootdir",
                str(project),
                *tests,
            ],
            project,
        )
        if result.returncode:
            return {
                "name": "Synthetic structure",
                "category": "correctness",
                "status": "failed",
                "detail": f"{label} tiny random-weight contract failed",
            }
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
    max_tokens = configuration.get("max_tokens", 1)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("checkpoint token count is invalid")
    encoded = json.dumps(configuration, separators=(",", ":"))
    observations = {"base": [], "head": []}
    for label, project in (
        ("base", base),
        ("head", head),
        ("head", head),
        ("base", base),
    ):
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
        observations[label].append(json.loads(result.stdout.strip().splitlines()[-1]))
    base_results, head_results = observations["base"], observations["head"]
    metric_specs = {
        "items_per_second": ("items/s", True),
        "prompt_tokens": ("tokens", False),
        "prefill_tps": ("tok/s", True),
        "decode_tps": ("tok/s", True),
        "ttft_ms": ("ms", False),
        "wall_ms": ("ms", False),
        "peak_memory_gib": ("GiB", False),
    }
    names = set(base_results[0]["metrics"]) & set(head_results[0]["metrics"])
    metrics = []
    for name in metric_specs:
        if name not in names:
            continue
        unit, higher_is_better = metric_specs[name]
        base_value = statistics.median(
            float(result["metrics"][name]) for result in base_results
        )
        head_value = statistics.median(
            float(result["metrics"][name]) for result in head_results
        )
        change = (
            0.0 if base_value == 0 else (head_value - base_value) / base_value * 100
        )
        paired_changes = []
        for base_result, head_result in zip(base_results, head_results):
            paired_base = float(base_result["metrics"][name])
            paired_head = float(head_result["metrics"][name])
            paired_changes.append(
                0.0
                if paired_base == 0
                else (paired_head - paired_base) / paired_base * 100
            )
        improved = all(
            paired >= 4 if higher_is_better else paired <= -4
            for paired in paired_changes
        )
        regressed = all(
            paired <= -4 if higher_is_better else paired >= 4
            for paired in paired_changes
        )
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
    base_signatures = {
        json.dumps(result["signature"], sort_keys=True) for result in base_results
    }
    head_signatures = {
        json.dumps(result["signature"], sort_keys=True) for result in head_results
    }
    match = (
        len(base_signatures) == len(head_signatures) == 1
        and base_signatures == head_signatures
    )
    if not match:
        for metric in metrics:
            metric["verdict"] = "advisory"
    return {
        "name": "Checkpoint output",
        "category": "correctness",
        "status": "passed" if match else "failed",
        "detail": (
            "Main and PR produced the same checkpoint output"
            if match
            else "Main and PR produced different checkpoint output"
        ),
    }, metrics


def _test_summary(result: subprocess.CompletedProcess[str]) -> str:
    stdout = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    stderr = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    summary = stdout[-1] if stdout else stderr[-1] if stderr else "no output"
    return " ".join(summary.split())[:70]


def _server_contract(
    job: Mapping[str, Any], control: Path, base: Path, head: Path
) -> dict[str, Any]:
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
        "--import-mode=importlib",
    ]
    test = str(control / selectors[0])
    base_result = _run([*command, "--rootdir", str(base), test], base)
    head_result = _run([*command, "--rootdir", str(head), test], head)
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
            check = _synthetic(job, control, base, head)
            checks.append(check)
            if check["status"] != "passed":
                break
        elif phase == "checkpoint":
            check, metrics = _checkpoint(job, control, base, head)
            checks.append(check)
        elif phase == "server_contract":
            checks.append(_server_contract(job, control, base, head))
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
