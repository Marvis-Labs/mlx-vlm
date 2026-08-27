"""Correctness gate for a newly added model: mlx against its reference.

A new model has no previous revision to compare against, so it is compared to
its implementation in the transformers library on identical weights and
identical inputs. Two quantities are reported: greedy agreement, the fraction
of positions where both pick the same next token, and the Kullback-Leibler
divergence of the mlx next-token distribution from the reference, mean and
max across positions -- which catches a shift the argmax survives.

Three constraints make the comparison meaningful, and violating any makes the
number measure the wrong thing:
  * unquantized weights -- quantization noise is larger than a port's error;
  * the reference in float32 on CPU -- an accelerated backend adds its own;
  * both models resident at once -- so this needs memory for two copies.

This is a separate module from the performance path on purpose: it needs
torch and transformers the benchmark environment does not carry, runs only on
a new model, and reports agreement rather than throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

os.environ.setdefault("MLX_ENABLE_TF32", "0")

PROMPT = (
    "The study of language reveals how humans encode meaning. Consider the "
    "sentence structure, the morphology, and the way context shapes "
    "interpretation across speakers and situations."
)


def _reference_logits(repo: str, token_ids):
    """Next-token log-probabilities from transformers, float32 on CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    with torch.no_grad():
        ids = torch.tensor([token_ids])
        logits = model(ids).logits[0]  # (seq, vocab)
        return torch.log_softmax(logits.float(), dim=-1).numpy()


def _mlx_logits(repo: str, token_ids):
    """Next-token log-probabilities from mlx-vlm, on the same tokens."""
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_vlm import load

    model, _ = load(repo)
    lm = getattr(model, "language_model", model)
    ids = mx.array([token_ids])
    out = lm(ids)
    # mlx-vlm returns a LanguageModelOutput; older/text models may return an
    # array or a tuple. Take .logits when present, else the raw output.
    logits = getattr(out, "logits", out)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits[0]  # (seq, vocab)
    return nn.log_softmax(logits.astype(mx.float32), axis=-1)


def compare(mlx_repo: str, ref_repo: str) -> Dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref_repo, trust_remote_code=True)
    token_ids = tok(PROMPT)["input_ids"]

    ref = np.asarray(_reference_logits(ref_repo, token_ids))
    got = np.asarray(_mlx_logits(mlx_repo, token_ids))

    n = min(len(ref), len(got))
    ref, got = ref[:n], got[:n]

    greedy = float((ref.argmax(-1) == got.argmax(-1)).mean())
    # KL(reference || mlx) per position, in nats.
    p = np.exp(ref)
    kl = (p * (ref - got)).sum(-1)
    return {
        "greedy_agreement": round(greedy, 4),
        "kl_mean": round(float(kl.mean()), 6),
        "kl_max": round(float(kl.max()), 6),
        "positions": int(n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-repo", required=True, help="the mlx-community checkpoint")
    ap.add_argument("--ref-repo", required=True, help="the transformers reference")
    ap.add_argument("--out", default="parity.json")
    args = ap.parse_args()
    try:
        result = compare(args.mlx_repo, args.ref_repo)
    except Exception as exc:  # a missing reference class, a load failure
        result = {"error": f"parity failed: {type(exc).__name__}: {exc}"}
    Path = __import__("pathlib").Path
    Path(args.out).write_text(json.dumps(result))
    print(json.dumps(result))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
