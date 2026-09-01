from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import CacheCharacteristic, ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.segmented import (
    MLXChunkedCacheAdapter,
    segmented_contract_cases,
)
from ci.kv_cache_profiles.dense import MLXDenseCacheAdapter
from mlx_vlm.models.cache import ChunkedKVCache, ConcatenateKVCache


def test_real_segmented_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in segmented_contract_cases()]

    assert [result.case for result in results] == [
        "ChunkedKVCache",
        "ConcatenateKVCache",
    ]
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_segmented_probe_reports_both_implementations():
    result = run(
        ("ci.kv_cache_profiles.segmented:segmented_contract_cases",)
    )

    assert result["verdict"] == "passed"
    assert result["profiles"] == ["segmented"]
    assert [case["case"] for case in result["cases"]] == [
        "ChunkedKVCache",
        "ConcatenateKVCache",
    ]


def test_segmented_cases_do_not_claim_backing_allocation_parity():
    for case in segmented_contract_cases():
        assert CacheCharacteristic.STORAGE not in case.characteristics


def test_chunked_contract_rejects_extra_front_eviction():
    class CorruptChunkedKVCache(ChunkedKVCache):
        def maybe_trim_front(self):
            super().maybe_trim_front()
            if self.start_position:
                self.keys = self.keys[..., 1:, :]
                self.values = self.values[..., 1:, :]
                self.start_position += 1

    original = segmented_contract_cases()[0]
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "front-eviction"
    )
    mutated = replace(
        original,
        name="CorruptChunkedKVCache",
        subject_factory=lambda: MLXChunkedCacheAdapter(
            lambda: CorruptChunkedKVCache(chunk_size=8)
        ),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].characteristic in {"content", "visibility"}


def test_chunked_contract_rejects_snapshot_offset_loss():
    class CorruptRestoreChunkedKVCache(ChunkedKVCache):
        def prefix_cache_restore(self, snapshot):
            self.state = snapshot["state"]
            values = tuple(map(int, snapshot["meta_state"]))
            self.chunk_size, self.start_position = values[:2]

    original = segmented_contract_cases()[0]
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "snapshot-after-front-eviction"
    )
    mutated = replace(
        original,
        name="CorruptRestoreChunkedKVCache",
        subject_factory=lambda: MLXChunkedCacheAdapter(
            lambda: CorruptRestoreChunkedKVCache(chunk_size=8)
        ),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert any(failure.characteristic == "position" for failure in result.failures)


def test_concatenate_contract_rejects_corrupted_append():
    class CorruptConcatenateKVCache(ConcatenateKVCache):
        def update_and_fetch(self, keys, values):
            cached_keys, cached_values = super().update_and_fetch(keys, values)
            self.values[..., -1, 0] = -1
            return cached_keys, cached_values

    original = segmented_contract_cases()[1]
    mutated = replace(
        original,
        name="CorruptConcatenateKVCache",
        subject_factory=lambda: MLXDenseCacheAdapter(
            CorruptConcatenateKVCache, original.capabilities
        ),
        sequences=(original.sequences[0],),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].characteristic == "content"
