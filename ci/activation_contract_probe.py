from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _append_check(
    checks: list[dict[str, Any]],
    outputs: list[float],
    name: str,
    actual: Any,
    expected: Any,
    tolerance: float,
) -> None:
    import mlx.core as mx
    import numpy as np

    mx.eval(actual, expected)
    actual_values = np.asarray(actual.astype(mx.float32))
    expected_values = np.asarray(expected.astype(mx.float32))
    difference = float(np.max(np.abs(actual_values - expected_values)))
    finite = bool(np.isfinite(actual_values).all())
    outputs.extend(actual_values.reshape(-1).tolist())
    checks.append(
        {
            "name": name,
            "shape": list(actual_values.shape),
            "max_abs_diff": difference,
            "tolerance": tolerance,
            "finite": finite,
            "match": finite
            and actual_values.shape == expected_values.shape
            and difference <= tolerance,
        }
    )


def _swiglu_reference(gate: Any, value: Any) -> Any:
    import mlx.core as mx

    return gate * mx.sigmoid(gate) * value


def _switch_reference(layer: Any, values: Any, indices: Any) -> Any:
    import mlx.core as mx

    batches = []
    for batch_index in range(values.shape[0]):
        sequence = []
        for token_index in range(values.shape[1]):
            experts = []
            for route_index in range(indices.shape[2]):
                expert = int(indices[batch_index, token_index, route_index].item())
                token = values[batch_index, token_index]
                gate = layer.gate_proj.weight[expert] @ token
                up = layer.up_proj.weight[expert] @ token
                if "bias" in layer.gate_proj:
                    gate = gate + layer.gate_proj.bias[expert]
                    up = up + layer.up_proj.bias[expert]
                hidden = _swiglu_reference(gate, up)
                output = layer.down_proj.weight[expert] @ hidden
                if "bias" in layer.down_proj:
                    output = output + layer.down_proj.bias[expert]
                experts.append(output)
            sequence.append(mx.stack(experts))
        batches.append(mx.stack(sequence))
    return mx.stack(batches)


def _swiglu_contract() -> dict[str, Any]:
    import mlx.core as mx

    from mlx_vlm.models.activations import swiglu
    from mlx_vlm.models.mlp import SwiGLUMLP
    from mlx_vlm.models.switch_layers import SwitchGLU

    checks: list[dict[str, Any]] = []
    outputs: list[float] = []
    cases = (
        ("vector-fp32", (17,), (17,), mx.float32, 1e-6),
        ("sequence-fp32", (2, 3, 8), (2, 3, 8), mx.float32, 1e-6),
        ("broadcast-fp32", (2, 3, 1), (2, 3, 8), mx.float32, 1e-6),
        ("sequence-fp16", (2, 5, 8), (2, 5, 8), mx.float16, 2e-3),
        ("sequence-bf16", (2, 5, 8), (2, 5, 8), mx.bfloat16, 2e-2),
    )
    for name, gate_shape, value_shape, dtype, tolerance in cases:
        gate_size = 1
        for dimension in gate_shape:
            gate_size *= dimension
        value_size = 1
        for dimension in value_shape:
            value_size *= dimension
        gate = (
            mx.arange(gate_size, dtype=mx.float32).reshape(gate_shape)
            / max(1, gate_size - 1)
            * 12
            - 6
        ).astype(dtype)
        value = (
            mx.arange(value_size, dtype=mx.float32).reshape(value_shape)
            / max(1, value_size - 1)
            * 4
            - 2
        ).astype(dtype)
        _append_check(
            checks,
            outputs,
            name,
            swiglu(gate, value),
            _swiglu_reference(gate, value),
            tolerance,
        )

    extreme_gate = mx.array([-40.0, -8.0, -0.0, 0.0, 8.0, 40.0])
    extreme_value = mx.array([0.0, 2.0, -3.0, 4.0, -2.0, 0.0])
    _append_check(
        checks,
        outputs,
        "zero-and-extreme-values",
        swiglu(extreme_gate, extreme_value),
        _swiglu_reference(extreme_gate, extreme_value),
        1e-6,
    )

    gate = mx.array([-5.0, -1.0, -0.25, 0.25, 1.5, 5.0])
    value = mx.array([2.0, -3.0, 0.5, -1.5, 4.0, -0.25])
    sigmoid = mx.sigmoid(gate)
    gate_gradient = mx.grad(lambda current: mx.sum(swiglu(current, value)))(gate)
    gate_reference = value * sigmoid * (1 + gate * (1 - sigmoid))
    _append_check(
        checks,
        outputs,
        "gate-gradient",
        gate_gradient,
        gate_reference,
        1e-5,
    )
    value_gradient = mx.grad(lambda current: mx.sum(swiglu(gate, current)))(value)
    value_reference = gate * sigmoid
    _append_check(
        checks,
        outputs,
        "value-gradient",
        value_gradient,
        value_reference,
        1e-5,
    )

    mx.random.seed(17)
    mlp = SwiGLUMLP(8, 16, bias=True)
    mlp_values = mx.arange(112, dtype=mx.float32).reshape(2, 7, 8) / 111 - 0.5
    mlp_output = mlp(mlp_values)
    mlp_reference = mlp.down_proj(
        _swiglu_reference(
            mlp.gate_proj(mlp_values),
            mlp.up_proj(mlp_values),
        )
    )
    _append_check(
        checks,
        outputs,
        "SwiGLUMLP-integration",
        mlp_output,
        mlp_reference,
        1e-5,
    )

    for name, shape, tolerance in (
        ("SwitchGLU-unsorted-integration", (2, 3, 2), 1e-5),
        ("SwitchGLU-sorted-integration", (1, 32, 2), 3e-3),
    ):
        mx.random.seed(23)
        switch = SwitchGLU(8, 12, 5, bias=True)
        size = shape[0] * shape[1] * 8
        switch_values = (
            mx.arange(size, dtype=mx.float32).reshape(shape[:2] + (8,))
            / max(1, size - 1)
            - 0.5
        )
        indices = (mx.arange(shape[0] * shape[1] * shape[2]).reshape(shape) % 5).astype(
            mx.uint32
        )
        _append_check(
            checks,
            outputs,
            name,
            switch(switch_values, indices),
            _switch_reference(switch, switch_values, indices),
            tolerance,
        )

    passed = bool(checks) and all(check["match"] for check in checks)
    return {
        "profile": "swiglu",
        "verdict": "passed" if passed else "test_failure",
        "checks": checks,
        "summary": {
            "checks": len(checks),
            "failures": [check["name"] for check in checks if not check["match"]],
        },
        "output_hash": hashlib.sha256(
            json.dumps(outputs, separators=(",", ":")).encode()
        ).hexdigest()[:16],
    }


def _softplus_reference(value: Any) -> Any:
    import mlx.core as mx

    return mx.log1p(mx.exp(-mx.abs(value))) + mx.maximum(value, 0)


def _xielu_reference(
    values: Any,
    alpha_p: Any,
    alpha_n: Any,
    beta: Any,
    eps: Any,
) -> Any:
    import mlx.core as mx

    positive_scale = _softplus_reference(alpha_p)
    negative_scale = beta + _softplus_reference(alpha_n)
    return mx.where(
        values > 0,
        positive_scale * mx.square(values) + beta * values,
        (mx.expm1(mx.minimum(values, eps)) - values) * negative_scale + beta * values,
    )


def _xielu_contract() -> dict[str, Any]:
    import mlx.core as mx

    from mlx_vlm.models.activations import XieLU, xielu
    from mlx_vlm.models.apertus.language import ApertusMLP

    checks: list[dict[str, Any]] = []
    outputs: list[float] = []
    alpha_p = mx.array(0.35)
    alpha_n = mx.array(-0.2)
    beta = mx.array(0.5)
    eps = mx.array(-1e-6)
    boundary = mx.array([-20.0, -3.0, -0.25, -1e-5, -1e-6, -1e-7, 0.0, 1e-7, 0.1, 3.0])
    for name, values, tolerance in (
        ("piecewise-boundary-fp32", boundary, 1e-6),
        ("piecewise-batch-fp32", boundary.reshape(2, 5), 1e-6),
        ("piecewise-fp16", boundary.astype(mx.float16), 3e-3),
        ("piecewise-bf16", boundary.astype(mx.bfloat16), 3e-2),
    ):
        _append_check(
            checks,
            outputs,
            name,
            xielu(values, alpha_p, alpha_n, beta, eps),
            _xielu_reference(values, alpha_p, alpha_n, beta, eps),
            tolerance,
        )

    module = XieLU(alpha_p_init=0.8, alpha_n_init=0.9, beta=0.5, eps=-1e-6)
    _append_check(
        checks,
        outputs,
        "module-integration",
        module(boundary),
        _xielu_reference(
            boundary,
            module.alpha_p,
            module.alpha_n,
            module.beta,
            module.eps,
        ),
        1e-6,
    )
    _append_check(
        checks,
        outputs,
        "alpha-p-initialization",
        module.alpha_p,
        mx.log(mx.exp(mx.array(0.8)) - 1),
        1e-6,
    )
    _append_check(
        checks,
        outputs,
        "alpha-n-initialization",
        module.alpha_n,
        mx.log(mx.exp(mx.array(0.4)) - 1),
        1e-6,
    )

    mx.random.seed(31)
    aperture = ApertusMLP(
        SimpleNamespace(hidden_size=8, intermediate_size=12, mlp_bias=True)
    )
    aperture_values = mx.arange(80, dtype=mx.float32).reshape(2, 5, 8) / 79 - 0.5
    aperture_hidden = aperture.up_proj(aperture_values)
    aperture_reference = aperture.down_proj(
        _xielu_reference(
            aperture_hidden,
            aperture.act_fn.alpha_p,
            aperture.act_fn.alpha_n,
            aperture.act_fn.beta,
            aperture.act_fn.eps,
        )
    )
    _append_check(
        checks,
        outputs,
        "ApertusMLP-integration",
        aperture(aperture_values),
        aperture_reference,
        1e-5,
    )

    gradient_values = mx.array([-2.0, -0.25, 0.25, 2.0])
    input_gradient = mx.grad(
        lambda current: mx.sum(xielu(current, alpha_p, alpha_n, beta, eps))
    )(gradient_values)
    positive_scale = _softplus_reference(alpha_p)
    negative_scale = beta + _softplus_reference(alpha_n)
    input_reference = mx.where(
        gradient_values > 0,
        2 * positive_scale * gradient_values + beta,
        (mx.exp(gradient_values) - 1) * negative_scale + beta,
    )
    _append_check(
        checks,
        outputs,
        "input-gradient",
        input_gradient,
        input_reference,
        1e-5,
    )
    alpha_p_gradient = mx.grad(
        lambda current: mx.sum(xielu(gradient_values, current, alpha_n, beta, eps))
    )(alpha_p)
    alpha_p_reference = mx.sum(
        mx.where(
            gradient_values > 0,
            mx.sigmoid(alpha_p) * mx.square(gradient_values),
            0,
        )
    )
    _append_check(
        checks,
        outputs,
        "alpha-p-gradient",
        alpha_p_gradient,
        alpha_p_reference,
        1e-5,
    )
    negative_term = mx.expm1(mx.minimum(gradient_values, eps)) - gradient_values
    alpha_n_gradient = mx.grad(
        lambda current: mx.sum(xielu(gradient_values, alpha_p, current, beta, eps))
    )(alpha_n)
    alpha_n_reference = mx.sum(
        mx.where(
            gradient_values > 0,
            0,
            negative_term * mx.sigmoid(alpha_n),
        )
    )
    _append_check(
        checks,
        outputs,
        "alpha-n-gradient",
        alpha_n_gradient,
        alpha_n_reference,
        1e-5,
    )

    passed = bool(checks) and all(check["match"] for check in checks)
    return {
        "profile": "xielu",
        "verdict": "passed" if passed else "test_failure",
        "checks": checks,
        "summary": {
            "checks": len(checks),
            "failures": [check["name"] for check in checks if not check["match"]],
        },
        "output_hash": hashlib.sha256(
            json.dumps(outputs, separators=(",", ":")).encode()
        ).hexdigest()[:16],
    }


def run(job: Mapping[str, Any]) -> dict[str, Any]:
    contract = job.get("activation_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("ActivationChange work has no activation contract")
    profile = str(contract.get("profile", ""))
    contracts = {
        "swiglu": _swiglu_contract,
        "xielu": _xielu_contract,
    }
    selected = contracts.get(profile)
    if selected is None:
        raise ValueError(f"unsupported activation profile: {profile}")
    return selected()


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
