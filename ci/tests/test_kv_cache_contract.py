from dataclasses import replace

from ci.kv_cache_contract import (
    OPERATION_CAPABILITIES,
    CacheCapability,
    CacheCharacteristic,
    CacheContractCase,
    CacheOperation,
    CacheOperationKind,
    ContractRunner,
    OperationSequence,
    StorageProfile,
    Tolerance,
    compare_values,
)
from ci.kv_cache_oracles import DenseKVOracle


def update(start, count, batch_size=1, dtype="float32"):
    keys = [
        [[[start + position] for position in range(count)]] for _ in range(batch_size)
    ]
    values = [
        [[[100 + start + position] for position in range(count)]]
        for _ in range(batch_size)
    ]
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {"keys": keys, "values": values, "dtype": dtype},
    )


def test_dense_oracle_appends_trims_and_restores():
    oracle = DenseKVOracle()

    first = oracle.apply(update(0, 3))
    oracle.apply(CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}))
    oracle.apply(update(3, 2))
    trimmed = oracle.apply(CacheOperation(CacheOperationKind.TRIM, {"count": 1}))
    restored = oracle.apply(
        CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"})
    )

    assert first.visible_positions == (0, 1, 2)
    assert trimmed.visible_positions == (0, 1, 2, 3)
    assert trimmed.logical_keys == ((((0,), (1,), (2,), (3,)),),)
    assert restored == first


def test_dense_oracle_filters_and_extracts_batch_rows():
    oracle = DenseKVOracle()
    keys = [[[[batch * 10 + position] for position in range(2)]] for batch in range(3)]
    values = [
        [[[100 + batch * 10 + position] for position in range(2)]] for batch in range(3)
    ]
    oracle.apply(
        CacheOperation(
            CacheOperationKind.UPDATE,
            {"keys": keys, "values": values, "dtype": "float16"},
        )
    )

    filtered = oracle.apply(
        CacheOperation(CacheOperationKind.FILTER, {"indices": [2, 0]})
    )
    extracted = oracle.apply(CacheOperation(CacheOperationKind.EXTRACT, {"index": 1}))

    assert filtered.batch_size == 2
    assert filtered.logical_keys[0][0] == ((20,), (21,))
    assert extracted.batch_size == 1
    assert extracted.logical_keys[0][0] == ((0,), (1,))
    assert extracted.dtype == ("float16", "float16")


class DenseSubject:
    capabilities = DenseKVOracle.capabilities

    def __init__(self, mutation=None):
        self.oracle = DenseKVOracle()
        self.mutation = mutation

    def apply(self, operation):
        observation = self.oracle.apply(operation)
        return self.mutation(observation) if self.mutation else observation


def dense_case(subject_factory=lambda: DenseSubject(), operations=None, **kwargs):
    characteristics = frozenset(
        {
            CacheCharacteristic.CONTENT,
            CacheCharacteristic.VISIBILITY,
            CacheCharacteristic.POSITION,
            CacheCharacteristic.SHAPE,
            CacheCharacteristic.DTYPE,
            CacheCharacteristic.BATCH_LAYOUT,
        }
    )
    return CacheContractCase(
        name="dense-test",
        profile=StorageProfile.DENSE,
        subject_factory=subject_factory,
        oracle_factory=DenseKVOracle,
        capabilities=DenseKVOracle.capabilities,
        characteristics=characteristics,
        sequences=(
            OperationSequence(
                "append-and-trim",
                tuple(
                    operations
                    or [
                        update(0, 2),
                        update(2, 2),
                        CacheOperation(CacheOperationKind.TRIM, {"count": 1}),
                    ]
                ),
            ),
        ),
        **kwargs,
    )


def test_contract_runner_accepts_operation_by_operation_parity():
    result = ContractRunner().run(dense_case())

    assert result.passed
    assert result.checks == 18
    assert result.to_dict()["verdict"] == "passed"
    assert result.to_dict()["runs"] == [
        {
            "sequence": "append-and-trim",
            "verdict": "passed",
            "operations": 3,
            "executed_operations": 3,
            "checks": 18,
            "failures": 0,
        }
    ]


def test_contract_runner_identifies_the_mutated_characteristic_and_step():
    calls = 0

    def shift_position(observation):
        nonlocal calls
        calls += 1
        if calls != 2:
            return observation
        return replace(observation, offset=observation.offset + 1)

    result = ContractRunner().run(
        dense_case(subject_factory=lambda: DenseSubject(shift_position))
    )

    assert not result.passed
    failure = result.failures[0]
    assert failure.sequence == "append-and-trim"
    assert failure.step == 1
    assert failure.operation == "update"
    assert failure.characteristic == "position"


def test_contract_runner_rejects_an_undeclared_operation():
    case = dense_case(
        operations=[CacheOperation(CacheOperationKind.QUANTIZE)],
    )

    result = ContractRunner().run(case)

    assert not result.passed
    assert result.checks == 0
    assert result.runs[0].executed_operations == 1
    assert (
        result.failures[0].reason
        == "unsupported_operation:quantize:case,subject,oracle"
    )


def test_recursive_comparison_honors_numeric_tolerance():
    tolerance = Tolerance(absolute=1e-3)

    assert compare_values({"x": [1.0]}, {"x": [1.0005]}, tolerance) is None
    assert (
        compare_values({"x": [1.0]}, {"x": [1.01]}, tolerance) == "mismatch:value.x[0]"
    )


def test_dense_oracle_rejects_incompatible_updates():
    oracle = DenseKVOracle()
    oracle.apply(update(0, 2, dtype="float32"))

    try:
        oracle.apply(update(2, 1, dtype="float16"))
        assert False
    except ValueError as error:
        assert str(error) == "update dtype differs from cached dtype"


def test_extend_has_an_explicit_contract_capability():
    assert OPERATION_CAPABILITIES[CacheOperationKind.EXTEND] is CacheCapability.EXTEND
