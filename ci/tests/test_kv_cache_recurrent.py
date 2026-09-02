from dataclasses import replace

import pytest

pytest.importorskip("mlx.core")

from ci.kv_cache_contract import ContractRunner
from ci.kv_cache_contract_probe import run
from ci.kv_cache_profiles.recurrent import (
    MLXArraysCacheAdapter,
    recurrent_contract_cases,
)
from mlx_vlm.models.cache import ArraysCache


def test_real_recurrent_cache_contract_passes():
    result = ContractRunner().run(recurrent_contract_cases()[0])

    assert result.passed, result.to_dict()
    assert [run.sequence for run in result.runs] == [
        "partial-slots-filter-extract",
        "batch-lifecycle",
        "sparse-slot-merge",
        "reset-and-reuse",
    ]


def test_recurrent_probe_reports_arrays_cache():
    result = run(("ci.kv_cache_profiles.recurrent:recurrent_contract_cases",))

    assert result["verdict"] == "passed"
    assert result["profiles"] == ["recurrent"]
    assert [case["case"] for case in result["cases"]] == ["ArraysCache"]


def test_recurrent_contract_rejects_first_slot_only_empty_check():
    class CorruptEmptyArraysCache(ArraysCache):
        def empty(self):
            return self.cache[0] is None

    original = recurrent_contract_cases()[0]
    sequence = next(
        sequence
        for sequence in original.sequences
        if sequence.name == "partial-slots-filter-extract"
    )
    mutated = replace(
        original,
        name="CorruptEmptyArraysCache",
        subject_factory=lambda: _adapter(CorruptEmptyArraysCache),
        sequences=(sequence,),
    )

    result = ContractRunner().run(mutated)

    assert not result.passed
    assert result.failures[0].characteristic == "metadata"


def _adapter(cache_type):
    adapter = MLXArraysCacheAdapter(3)
    adapter.cache = cache_type(3)
    return adapter
