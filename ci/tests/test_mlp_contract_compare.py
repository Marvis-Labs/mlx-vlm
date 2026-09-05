from ci.mlp_contract_compare import compare


def probe(output, reference_diff=0.0, signature="same"):
    return {
        "symbols": {
            "SwiGLUMLP": {
                "output": output,
                "output_shape": [1, len(output)],
                "output_hash": "hash",
                "parameter_signature": signature,
                "finite": True,
                "reference_max_abs_diff": reference_diff,
            }
        }
    }


def test_mlp_contract_requires_base_head_and_reference_parity():
    result = compare(probe([1.0, 2.0]), probe([1.0, 2.000001]))

    assert result["verdict"] == "passed"
    assert result["correctness"]["match"] is True


def test_mlp_contract_rejects_matching_base_head_when_reference_is_wrong():
    result = compare(
        probe([1.0, 2.0], reference_diff=0.1),
        probe([1.0, 2.0], reference_diff=0.1),
    )

    assert result["verdict"] == "test_failure"
    assert result["correctness"]["match"] is False
