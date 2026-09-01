from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_profiles.windowed import (
    MLXWindowedCacheAdapter,
    windowed_contract_cases,
)
from mlx_vlm.models.cache import RotatingKVCache


def test_real_windowed_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in windowed_contract_cases()]

    assert [result.case for result in results] == [
        "RotatingKVCache",
        "BufferedRotatingKVCache",
    ]
    assert sum(result.checks for result in results) == 2002
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_windowed_contract_rejects_corrupted_rotating_content():
    class CorruptRotatingKVCache(RotatingKVCache):
        def update_and_fetch(self, keys, values):
            fetched = super().update_and_fetch(keys, values)
            self.values[..., self._idx - 1, 0] = -1
            return fetched

    original = windowed_contract_cases()[0]
    mutated = replace(
        original,
        name="CorruptRotatingKVCache",
        subject_factory=lambda: MLXWindowedCacheAdapter(
            lambda: CorruptRotatingKVCache(max_size=8, keep=2)
        ),
        sequences=(original.sequences[0],),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].sequence == "token-wrap"
    assert result.failures[0].characteristic == "content"
