from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_batch import MLXBatchKVAdapter
from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_profiles.windowed import (
    MLXWindowedCacheAdapter,
    windowed_contract_cases,
)
from mlx_vlm.models.cache import BatchRotatingKVCache, RotatingKVCache


def test_real_windowed_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in windowed_contract_cases()]

    assert [result.case for result in results] == [
        "RotatingKVCache",
        "BufferedRotatingKVCache",
        "BatchRotatingKVCache",
    ]
    assert [len(result.runs) for result in results] == [8, 8, 13]
    assert sum(result.checks for result in results) == 3087
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


def test_windowed_batch_contract_rejects_a_broken_finalize():
    class CorruptBatchRotatingKVCache(BatchRotatingKVCache):
        def finalize(self):
            return None

    original = next(
        case
        for case in windowed_contract_cases()
        if case.name == "BatchRotatingKVCache"
    )
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "right-padding-finalize-decode"
    )
    mutated = replace(
        original,
        name="CorruptBatchRotatingKVCache",
        subject_factory=lambda: MLXBatchKVAdapter(
            lambda left_padding: CorruptBatchRotatingKVCache(8, list(left_padding)),
            lambda: RotatingKVCache(8),
            BatchRotatingKVCache,
            profile=original.profile,
            batch_size=3,
            max_size=8,
        ),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].operation == "finalize_batch"
    assert result.failures[0].characteristic in {"content", "position"}
