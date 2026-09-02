from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _instance(symbol: str) -> Any:
    import mlx.core as mx

    from mlx_vlm.models.mlp import (
        GELUMLP,
        DeepseekMLP,
        FastGELUMLP,
        SwiGLUMLP,
        TanhGELUMLP,
    )

    config = SimpleNamespace(hidden_size=8, intermediate_size=16)
    constructors = {
        "SwiGLUMLP": lambda: SwiGLUMLP(8, 16, bias=True),
        "DeepseekMLP": lambda: DeepseekMLP(config),
        "GELUMLP": lambda: GELUMLP(config),
        "FastGELUMLP": lambda: FastGELUMLP(config),
        "TanhGELUMLP": lambda: TanhGELUMLP(config),
    }
    constructor = constructors.get(symbol)
    if constructor is None:
        raise ValueError(f"unsupported MLP symbol: {symbol}")
    mx.random.seed(0)
    return constructor()


def _reference(symbol: str, model: Any, values: Any) -> Any:
    import mlx.nn as nn

    from mlx_vlm.models.activations import swiglu

    if symbol in {"SwiGLUMLP", "DeepseekMLP"}:
        return model.down_proj(swiglu(model.gate_proj(values), model.up_proj(values)))
    activation = {
        "GELUMLP": nn.GELU(approx="precise"),
        "FastGELUMLP": nn.GELU(approx="fast"),
        "TanhGELUMLP": nn.GELU(approx="tanh"),
    }[symbol]
    return model.fc2(activation(model.fc1(values)))


def run(job: Mapping[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    contract = job.get("mlp_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("ModelPath work has no MLP contract")
    symbols = contract.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("MLP contract has no symbols")

    results: dict[str, Any] = {}
    for symbol in symbols:
        model = _instance(str(symbol))
        outputs: list[float] = []
        shapes: list[list[int]] = []
        reference_diff = 0.0
        cases = (
            ((1, 1, 8), mx.float32),
            ((1, 16, 8), mx.float32),
            ((2, 7, 8), mx.float32),
            ((1, 5, 8), mx.float16),
        )
        for shape, dtype in cases:
            size = shape[0] * shape[1] * shape[2]
            values = mx.arange(size, dtype=dtype).reshape(shape) / max(1, size - 1)
            output = model(values)
            reference = _reference(str(symbol), model, values)
            mx.eval(output, reference)
            array = np.asarray(output.astype(mx.float32))
            expected = np.asarray(reference.astype(mx.float32))
            outputs.extend(array.reshape(-1).tolist())
            shapes.append(list(array.shape))
            reference_diff = max(
                reference_diff, float(np.max(np.abs(array - expected)))
            )
        mx.eval(model.parameters())
        parameters = [
            (name, tuple(int(dimension) for dimension in value.shape))
            for name, value in tree_flatten(model.parameters())
        ]
        results[str(symbol)] = {
            "output": outputs,
            "output_shape": shapes,
            "output_hash": hashlib.sha256(
                np.asarray(outputs, dtype=np.float32).tobytes()
            ).hexdigest()[:16],
            "parameter_signature": hashlib.sha256(
                json.dumps(parameters, separators=(",", ":")).encode()
            ).hexdigest()[:16],
            "finite": bool(np.isfinite(outputs).all()),
            "reference_max_abs_diff": reference_diff,
        }
    return {"consumer": contract.get("consumer"), "symbols": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text())
    result = run(job)
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
