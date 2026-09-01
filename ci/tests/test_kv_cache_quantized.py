from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.quantized import (
    MLXQuantizedCacheAdapter,
    quantized_contract_cases,
)
from mlx_vlm.models.cache import QuantizedKVCache


def test_real_quantized_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in quantized_contract_cases()]

    assert [result.case for result in results] == [
        "QuantizedKVCache",
        "BatchQuantizedKVCache",
    ]
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_quantized_probe_reports_single_and_batch_implementations():
    result = run(("ci.kv_cache_profiles.quantized:quantized_contract_cases",))

    assert result["verdict"] == "passed"
    assert result["profiles"] == ["quantized"]
    assert [case["case"] for case in result["cases"]] == [
        "QuantizedKVCache",
        "BatchQuantizedKVCache",
    ]


def test_quantized_contract_rejects_corrupted_dequantized_content():
    class CorruptQuantizedKVCache(QuantizedKVCache):
        def dequantize_for_apc(self):
            keys, values = super().dequantize_for_apc()
            if values is not None:
                values = values + 1
            return keys, values

    original = quantized_contract_cases()[0]
    mutated = replace(
        original,
        name="CorruptQuantizedKVCache",
        subject_factory=lambda: MLXQuantizedCacheAdapter(
            lambda: CorruptQuantizedKVCache(group_size=32, bits=8),
            original.capabilities,
        ),
        sequences=(original.sequences[0],),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].characteristic == "content"
