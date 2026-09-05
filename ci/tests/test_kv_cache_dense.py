from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_batch import MLXBatchKVAdapter
from ci.kv_cache_contract import CacheCharacteristic, ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.common import MLXDenseCacheAdapter
from ci.kv_cache_profiles.dense import dense_contract_cases
from mlx_vlm.models.cache import BatchKVCache, KVCache, SimpleKVCache


def test_real_dense_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in dense_contract_cases()]

    assert [result.case for result in results] == [
        "KVCache",
        "SimpleKVCache",
        "BatchKVCache",
    ]
    assert [len(result.runs) for result in results] == [10, 8, 9]
    assert sum(result.checks for result in results) == 2669
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_dense_cases_do_not_claim_storage_parity():
    for case in dense_contract_cases():
        assert CacheCharacteristic.STORAGE not in case.characteristics


def test_dense_probe_reports_all_implementations():
    result = run()

    assert result["verdict"] == "passed"
    assert result["checks"] == 2669
    assert [case["case"] for case in result["cases"]] == [
        "KVCache",
        "SimpleKVCache",
        "BatchKVCache",
    ]
    assert [run["sequence"] for run in result["cases"][0]["runs"]] == [
        "append-trim-resume",
        "snapshot-restore-resume",
        "batch-row-extraction",
        "allocation-boundary",
        "reset-and-reuse",
        "kv-state-machine-0",
        "kv-state-machine-1",
        "kv-state-machine-2",
        "kv-state-machine-3",
        "kv-state-machine-4",
    ]


def test_dense_contract_rejects_a_real_cache_content_mutation():
    class CorruptSimpleKVCache(SimpleKVCache):
        def update_and_fetch(self, keys, values):
            cached_keys, cached_values = super().update_and_fetch(keys, values)
            self.values[..., -1, 0] = -1
            return cached_keys, cached_values

    original = dense_contract_cases()[1]
    mutated = replace(
        original,
        name="CorruptSimpleKVCache",
        subject_factory=lambda: MLXDenseCacheAdapter(
            CorruptSimpleKVCache, original.capabilities
        ),
        sequences=(original.sequences[0],),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].sequence == "append"
    assert result.failures[0].step == 0
    assert result.failures[0].characteristic == "content"


class CorruptTrimKVCache(KVCache):
    def trim(self, n):
        return min(self.offset, n)


class CorruptRestoreKVCache(KVCache):
    def prefix_cache_restore(self, snapshot):
        super().prefix_cache_restore(snapshot)
        self.values[..., 0, 0] = -1


class CorruptExtractKVCache(KVCache):
    def extract(self, idx):
        return super().extract(0)


class CorruptDtypeSimpleKVCache(SimpleKVCache):
    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        return super().update_and_fetch(
            keys.astype(mx.float16), values.astype(mx.float16)
        )


@pytest.mark.parametrize(
    ("cache_type", "sequence_name", "characteristic"),
    [
        (CorruptTrimKVCache, "append-trim-resume", "content"),
        (CorruptRestoreKVCache, "snapshot-restore-resume", "content"),
        (CorruptExtractKVCache, "batch-row-extraction", "content"),
        (CorruptDtypeSimpleKVCache, "append", "dtype"),
    ],
)
def test_dense_contract_rejects_lifecycle_mutations(
    cache_type, sequence_name, characteristic
):
    case_name = "SimpleKVCache" if issubclass(cache_type, SimpleKVCache) else "KVCache"
    original = next(case for case in dense_contract_cases() if case.name == case_name)
    sequence = next(
        sequence for sequence in original.sequences if sequence.name == sequence_name
    )
    mutated = replace(
        original,
        name=cache_type.__name__,
        subject_factory=lambda: MLXDenseCacheAdapter(cache_type, original.capabilities),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].sequence == sequence_name
    assert result.failures[0].characteristic == characteristic


def test_dense_batch_contract_rejects_a_broken_finalize():
    class CorruptBatchKVCache(BatchKVCache):
        def finalize(self):
            return None

    original = next(
        case for case in dense_contract_cases() if case.name == "BatchKVCache"
    )
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "padding-finalize-filter-extract"
    )
    mutated = replace(
        original,
        name="CorruptBatchKVCache",
        subject_factory=lambda: MLXBatchKVAdapter(
            CorruptBatchKVCache,
            KVCache,
            BatchKVCache,
            profile=original.profile,
            batch_size=3,
        ),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].operation == "finalize_batch"
    assert result.failures[0].characteristic in {"content", "position"}
