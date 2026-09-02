from ci.activation_contract_probe import run


def job(profile):
    return {
        "activation_contract": {
            "profile": profile,
            "symbols": [profile],
            "oracle": "independent_mathematical_contract",
        }
    }


def test_shared_swiglu_satisfies_independent_contract():
    result = run(job("swiglu"))

    assert result["verdict"] == "passed"
    assert result["summary"] == {"checks": 11, "failures": []}
    assert {check["name"] for check in result["checks"]} >= {
        "gate-gradient",
        "value-gradient",
        "SwiGLUMLP-integration",
        "SwitchGLU-unsorted-integration",
        "SwitchGLU-sorted-integration",
    }


def test_shared_xielu_satisfies_independent_contract():
    result = run(job("xielu"))

    assert result["verdict"] == "passed"
    assert result["summary"] == {"checks": 11, "failures": []}
    assert {check["name"] for check in result["checks"]} >= {
        "piecewise-boundary-fp32",
        "module-integration",
        "ApertusMLP-integration",
        "input-gradient",
        "alpha-p-gradient",
        "alpha-n-gradient",
    }


def test_swiglu_contract_rejects_an_incorrect_gate_formula(monkeypatch):
    import mlx_vlm.models.activations as activations

    monkeypatch.setattr(activations, "swiglu", lambda gate, value: gate * value)

    result = run(job("swiglu"))

    assert result["verdict"] == "test_failure"
    assert result["summary"]["failures"]


def test_xielu_contract_rejects_an_incorrect_piecewise_function(monkeypatch):
    import mlx_vlm.models.activations as activations

    monkeypatch.setattr(
        activations,
        "xielu",
        lambda values, alpha_p, alpha_n, beta, eps: values,
    )

    result = run(job("xielu"))

    assert result["verdict"] == "test_failure"
    assert "piecewise-boundary-fp32" in result["summary"]["failures"]
