from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.pooling import MLXPoolingCacheAdapter, pooling_contract_cases
from mlx_vlm.models.cache import PoolingCache


def test_real_pooling_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in pooling_contract_cases()]

    assert [result.case for result in results] == [
        "PoolingCache",
        "BatchPoolingCache",
    ]
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_pooling_probe_reports_single_and_batch_implementations():
    result = run(("ci.kv_cache_profiles.pooling:pooling_contract_cases",))

    assert result["verdict"] == "passed"
    assert result["profiles"] == ["pooling"]
    assert [case["case"] for case in result["cases"]] == [
        "PoolingCache",
        "BatchPoolingCache",
    ]


def test_pooling_contract_rejects_corrupted_remainder():
    class CorruptPoolingCache(PoolingCache):
        def accumulate_windows(self, kv, gate, offset):
            result = super().accumulate_windows(kv, gate, offset)
            self.remainder = max(0, self.remainder - 1)
            return result

    original = pooling_contract_cases()[0]
    mutated = replace(
        original,
        name="CorruptPoolingCache",
        subject_factory=lambda: _adapter(CorruptPoolingCache),
        sequences=(original.sequences[0],),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].characteristic in {"content", "position"}


def _adapter(cache_type):
    adapter = MLXPoolingCacheAdapter(4)
    adapter.cache = cache_type(4)
    return adapter
