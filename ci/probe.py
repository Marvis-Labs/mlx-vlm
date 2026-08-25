"""Execute one measurement inside one revision of mlx-vlm.

Two revisions of the same package cannot coexist in a process, so the
orchestrator sets PYTHONPATH to a worktree and runs this as a subprocess.
Everything this writes to stdout is one JSON object; anything the engine
prints goes to stderr so it cannot corrupt the result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Determinism: the comparison is between two revisions, so anything that
# varies run to run has to be pinned before the engine is imported.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")

PROMPT_TOKENS_TARGET = 512
LONG_PROMPT_TOKENS_TARGET = 4096

# A fixed, boring prompt. Content is irrelevant; stability is not.
_STEM = (
    "Describe the process of photosynthesis in precise terms, covering the "
    "light-dependent reactions, the Calvin cycle, and the role of chlorophyll. "
)


def build_prompt(target_tokens: int) -> str:
    # Roughly four characters per token is close enough; the exact count is
    # recorded from the engine's own prompt_tokens.
    return (_STEM * max(1, target_tokens * 4 // len(_STEM)))[: target_tokens * 4]


def _mx():
    import mlx.core as mx

    return mx


def reset_device() -> None:
    """Drop cached buffers so peak memory means the same thing every run."""
    mx = _mx()
    gc.collect()
    for name in ("clear_cache", "reset_peak_memory"):
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        if callable(fn):
            fn()


def _finish(results: List[Any]) -> Dict[str, Any]:
    """Collapse a stream of GenerationResult into the metrics we keep."""
    last = results[-1]
    token_ids = [r.token for r in results if getattr(r, "token", None) is not None]
    digest = hashlib.sha256(",".join(str(t) for t in token_ids).encode()).hexdigest()[
        :16
    ]
    return {
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "prefill_tps": round(last.prompt_tps, 3),
        "decode_tps": round(last.generation_tps, 3),
        "peak_mem_gb": round(last.peak_memory, 4),
        "cached_tokens": getattr(last, "cached_tokens", 0),
        "output_hash": digest,
        "finish_reason": last.finish_reason,
    }


def _stream(model, processor, prompt: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from mlx_vlm import stream_generate

    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    collected = []
    for res in stream_generate(model, processor, prompt, **kwargs):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - started) * 1000.0
        collected.append(res)
    if not collected:
        raise RuntimeError("generation produced no results")
    out = _finish(collected)
    out["ttft_ms"] = round(ttft_ms or 0.0, 3)
    out["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return out


def apc_manager():
    """Build the prefix-cache manager this process will own.

    There is no module-level singleton: ``stream_generate`` takes an
    ``apc_manager`` kwarg. Constructing it here means the probe holds the
    object and can read its counters directly, rather than reaching for a
    global that does not exist.
    """
    import mlx_vlm.apc as apc

    return apc.from_env()


def apc_stats(manager) -> Dict[str, Any]:
    """Functional counters from the component under test.

    A change that holds throughput flat while the hit rate collapses is a
    regression timing alone cannot see, so these outrank the timings.
    """
    if manager is None:
        return {}
    try:
        snap = manager.stats.snapshot(manager.num_blocks, manager.block_size)
    except Exception:
        return {}
    keep = (
        "token_hit_rate",
        "matched_tokens",
        "served_tokens",
        "lookups_hit",
        "lookups_miss",
        "exact_hits",
        "rejects",
        "rejects_by_reason",
    )
    return {k: snap[k] for k in keep if k in snap}


# --------------------------------------------------------------- scenarios


def _kwargs(cell, manager) -> Dict[str, Any]:
    kw = dict(BASE_KWARGS, **(cell.get("args") or {}))
    if manager is not None:
        kw["apc_manager"] = manager
    return kw


def single_generation(model, processor, cell) -> Dict[str, Any]:
    manager = apc_manager()
    out = _stream(
        model, processor, build_prompt(PROMPT_TOKENS_TARGET), _kwargs(cell, manager)
    )
    out.update(apc_stats(manager))
    return out


def long_prompt(model, processor, cell) -> Dict[str, Any]:
    manager = apc_manager()
    out = _stream(
        model,
        processor,
        build_prompt(LONG_PROMPT_TOKENS_TARGET),
        _kwargs(cell, manager),
    )
    out.update(apc_stats(manager))
    return out


def shared_prefix_pair(model, processor, cell) -> Dict[str, Any]:
    """Prefix caching does no work on a first request.

    A single-request cell would report a change of zero regardless of what
    the diff did, so the first call only populates and the second is what
    gets measured.
    """
    prefix = build_prompt(PROMPT_TOKENS_TARGET)
    manager = apc_manager()
    kwargs = _kwargs(cell, manager)
    _stream(model, processor, prefix + " First question?", kwargs)  # prime
    measured = _stream(model, processor, prefix + " Second question?", kwargs)
    measured.update(apc_stats(manager))
    return measured


SCENARIOS = {
    "single_generation": single_generation,
    "long_prompt": long_prompt,
    "shared_prefix_pair": shared_prefix_pair,
}

BASE_KWARGS: Dict[str, Any] = {"max_tokens": 128, "temperature": 0.0, "verbose": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="path to the cell JSON")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    args = ap.parse_args()

    cell = json.loads(open(args.cell).read())
    scenario = SCENARIOS.get(cell["scenario"])
    if scenario is None:
        print(json.dumps({"error": f"unknown scenario {cell['scenario']}"}))
        return 2

    for key, value in (cell.get("env") or {}).items():
        os.environ[key] = str(value)

    from mlx_vlm import load

    model, processor = load(cell["repo"], revision=cell["revision"] or None)

    # Warmup is not optional: the first pass compiles Metal kernels, and
    # folding that into a measurement makes the first revision look slower.
    for _ in range(args.warmup):
        reset_device()
        scenario(model, processor, cell)

    runs = []
    for _ in range(args.iterations):
        reset_device()
        runs.append(scenario(model, processor, cell))

    print(
        json.dumps(
            {
                "cell": cell["id"],
                "revision_under_test": os.environ.get("CI_REVISION_LABEL", "?"),
                "runs": runs,
                "mlx_version": __import__(
                    "mlx.core", fromlist=["__version__"]
                ).__version__,
                "python": sys.version.split()[0],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
