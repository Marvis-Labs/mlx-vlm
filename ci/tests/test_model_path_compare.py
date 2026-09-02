from ci.model_path_compare import compare, metric_verdict


def measurements(**overrides):
    value = {
        "prefill_tps": 100.0,
        "decode_tps": 20.0,
        "ttft_ms": 500.0,
        "wall_ms": 1000.0,
        "peak_memory_gib": 4.0,
        "generation_tokens": 16,
        "output_hash": "same",
    }
    value.update(overrides)
    return value


def test_metric_direction_accounts_for_throughput_and_latency():
    assert metric_verdict("prefill_tps", 8) == "improved"
    assert metric_verdict("decode_tps", -8) == "regressed"
    assert metric_verdict("ttft_ms", -8) == "improved"
    assert metric_verdict("peak_memory_gib", 8) == "regressed"
    assert metric_verdict("wall_ms", 2) == "noise"


def test_comparison_reports_improvement_with_matching_output():
    result = compare(
        measurements(),
        measurements(prefill_tps=110.0, ttft_ms=450.0),
    )

    assert result["verdict"] == "improved"
    assert result["correctness"]["match"] is True
    assert result["metrics"]["prefill_tps"]["change_pct"] == 10.0


def test_output_mismatch_overrides_performance():
    result = compare(measurements(), measurements(output_hash="different"))

    assert result["verdict"] == "test_failure"
    assert result["correctness"]["match"] is False


def test_decode_throughput_is_unavailable_for_short_generations():
    result = compare(
        measurements(generation_tokens=1),
        measurements(generation_tokens=1, decode_tps=200.0),
    )

    assert result["verdict"] == "passed"
    assert "decode_tps" not in result["metrics"]
    assert result["unavailable_metrics"]["decode_tps"] == {
        "reason": "requires_at_least_8_generation_tokens",
        "base_generation_tokens": 1,
        "head_generation_tokens": 1,
    }
