from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def checkpoint(job: Mapping[str, Any]) -> tuple[str, str]:
    value = job.get("hf_checkpoint")
    if not isinstance(value, Mapping):
        raise ValueError("job has no hf_checkpoint")
    repo = value.get("repo")
    revision = value.get("revision")
    if not isinstance(repo, str) or not repo:
        raise ValueError("hf_checkpoint repo is required")
    if not isinstance(revision, str) or not revision:
        raise ValueError("hf_checkpoint revision is required")
    return repo, revision


def summarize(results: Sequence[Any], started: float, ttft_ms: float) -> dict[str, Any]:
    if not results:
        raise RuntimeError("generation produced no results")
    last = results[-1]
    text = "".join(str(getattr(item, "text", "")) for item in results).strip()
    if not text:
        raise RuntimeError("generation produced no text")
    token_ids = [
        int(item.token) for item in results if getattr(item, "token", None) is not None
    ]
    return {
        "generated_text": text,
        "output_hash": hashlib.sha256(
            ",".join(str(token) for token in token_ids).encode()
        ).hexdigest()[:16],
        "prompt_tokens": int(last.prompt_tokens),
        "generation_tokens": int(last.generation_tokens),
        "prefill_tps": round(float(last.prompt_tps), 3),
        "decode_tps": round(float(last.generation_tps), 3),
        "ttft_ms": round(ttft_ms, 3),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "peak_memory_gib": round(float(last.peak_memory), 4),
        "finish_reason": last.finish_reason,
    }


def run(
    job: Mapping[str, Any], image: Path, prompt: str, max_tokens: int
) -> dict[str, Any]:
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    repo, revision = checkpoint(job)
    model, processor = load(repo, revision=revision)
    config = load_config(repo, revision=revision)
    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    started = time.perf_counter()
    first_token_at = None
    results = []
    for result in stream_generate(
        model,
        processor,
        formatted,
        image=str(image),
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        results.append(result)

    findings = summarize(
        results,
        started,
        ((first_token_at or time.perf_counter()) - started) * 1000,
    )
    findings.update(
        {
            "model": repo,
            "revision": revision,
            "prompt": prompt,
            "image": image.name,
        }
    )
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Describe the animal in this image in one sentence."
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    findings = run(
        json.loads(args.job.read_text()), args.image, args.prompt, args.max_tokens
    )
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    print(json.dumps(findings, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
