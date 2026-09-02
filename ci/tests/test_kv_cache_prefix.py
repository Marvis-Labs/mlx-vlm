from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.prefix import MLXPrefixCacheAdapter, prefix_contract_cases
from mlx_vlm.models.cache import KVCache, StaticPrefixKVCache


def test_real_prefix_cache_contracts_pass():
    results = [ContractRunner().run(case) for case in prefix_contract_cases()]

    assert [result.case for result in results] == [
        "StaticPrefixKVCache",
        "KVCachePrefixProtocol",
    ]
    assert all(result.passed for result in results), [
        result.to_dict() for result in results
    ]


def test_prefix_probe_reports_static_and_dynamic_protocols():
    result = run(("ci.kv_cache_profiles.prefix:prefix_contract_cases",))

    assert result["verdict"] == "passed"
    assert result["profiles"] == ["prefix"]
    assert [case["case"] for case in result["cases"]] == [
        "StaticPrefixKVCache",
        "KVCachePrefixProtocol",
    ]


def test_prefix_contract_rejects_read_only_restore_mutation():
    class CorruptStaticPrefixKVCache(StaticPrefixKVCache):
        @property
        def meta_state(self):
            return tuple(map(str, (self.max_size, self.step, self.offset)))

        @meta_state.setter
        def meta_state(self, value):
            self.max_size, self.step, self.offset = map(int, value[:3])
            self.read_only = False

    original = prefix_contract_cases()[0]
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "read-only-snapshot-restore"
    )
    mutated = replace(
        original,
        name="CorruptStaticPrefixKVCache",
        subject_factory=lambda: MLXPrefixCacheAdapter(
            "static", lambda: CorruptStaticPrefixKVCache(max_size=8, step=4)
        ),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert any(
        failure.characteristic in {"content", "metadata"} for failure in result.failures
    )


def test_prefix_contract_rejects_missing_kv_reservation():
    class CorruptReserveKVCache(KVCache):
        def prefix_cache_reserve(self, min_capacity_tokens):
            return ()

    original = prefix_contract_cases()[1]
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "restore-reserve-continuation"
    )
    mutated = replace(
        original,
        name="CorruptReserveKVCache",
        subject_factory=lambda: MLXPrefixCacheAdapter("kv", CorruptReserveKVCache),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert any(failure.characteristic == "metadata" for failure in result.failures)
