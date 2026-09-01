from __future__ import annotations

import random
from typing import Any

from ci.kv_cache_batch import BATCH_CAPABILITIES, BatchKVOracle, MLXBatchKVAdapter
from ci.kv_cache_contract import (
    CacheCapability,
    CacheCharacteristic,
    CacheContractCase,
    CacheOperation,
    CacheOperationKind,
    OperationSequence,
    StorageProfile,
)
from ci.kv_cache_oracles import DenseKVOracle
from ci.kv_cache_profiles.common import MLXDenseCacheAdapter
from ci.kv_cache_profiles.common import cache_update as dense_update


def dense_contract_cases() -> tuple[CacheContractCase, ...]:
    from mlx_vlm.models.cache import BatchKVCache, KVCache, SimpleKVCache

    shared = frozenset(
        {
            CacheCapability.UPDATE,
            CacheCapability.RESET,
            CacheCapability.SNAPSHOT_RESTORE,
        }
    )
    kv_capabilities = shared | {CacheCapability.TRIM, CacheCapability.EXTRACT}
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
    return (
        CacheContractCase(
            name="KVCache",
            profile=StorageProfile.DENSE,
            subject_factory=lambda: MLXDenseCacheAdapter(KVCache, kv_capabilities),
            oracle_factory=DenseKVOracle,
            capabilities=kv_capabilities,
            characteristics=characteristics,
            sequences=_kv_cache_sequences() + _random_sequences("kv", allow_trim=True),
        ),
        CacheContractCase(
            name="SimpleKVCache",
            profile=StorageProfile.DENSE,
            subject_factory=lambda: MLXDenseCacheAdapter(SimpleKVCache, shared),
            oracle_factory=DenseKVOracle,
            capabilities=shared,
            characteristics=characteristics,
            sequences=_simple_kv_cache_sequences()
            + _random_sequences("simple", allow_trim=False),
        ),
        CacheContractCase(
            name="BatchKVCache",
            profile=StorageProfile.DENSE,
            subject_factory=lambda: MLXBatchKVAdapter(
                BatchKVCache,
                KVCache,
                BatchKVCache,
                profile=StorageProfile.DENSE,
                batch_size=3,
            ),
            oracle_factory=lambda: BatchKVOracle(3, profile=StorageProfile.DENSE),
            capabilities=BATCH_CAPABILITIES,
            characteristics=characteristics | {CacheCharacteristic.METADATA},
            sequences=_batch_sequences() + _batch_random_sequences(),
        ),
    )


def _kv_cache_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "append-trim-resume",
            (
                dense_update(0, 2),
                dense_update(2, 3),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                dense_update(3, 1),
            ),
        ),
        OperationSequence(
            "snapshot-restore-resume",
            (
                dense_update(0, 3, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                dense_update(3, 2, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                dense_update(3, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "batch-row-extraction",
            (
                dense_update(0, 2, batch_size=3),
                CacheOperation(CacheOperationKind.EXTRACT, {"index": 2}),
            ),
        ),
        OperationSequence(
            "allocation-boundary",
            (
                dense_update(0, 255, heads=1, key_channels=1, value_channels=1),
                dense_update(255, 2, heads=1, key_channels=1, value_channels=1),
                CacheOperation(CacheOperationKind.TRIM, {"count": 1}),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                dense_update(0, 2),
                CacheOperation(CacheOperationKind.RESET),
                dense_update(10, 1),
            ),
        ),
    )


def _simple_kv_cache_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "append",
            (
                dense_update(0, 1),
                dense_update(1, 3),
                dense_update(4, 2),
            ),
        ),
        OperationSequence(
            "snapshot-restore-resume",
            (
                dense_update(0, 3, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                dense_update(3, 2, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                dense_update(3, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                dense_update(0, 2),
                CacheOperation(CacheOperationKind.RESET),
                dense_update(10, 1),
            ),
        ),
    )


def _batch_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "padding-finalize-filter-extract",
            (
                CacheOperation(
                    CacheOperationKind.PREPARE_BATCH,
                    {
                        "left_padding": [2, 0, 1],
                        "right_padding": [1, 0, 2],
                        "lengths": [3, 6, 3],
                    },
                ),
                dense_update(0, 6, batch_size=3),
                CacheOperation(CacheOperationKind.FINALIZE_BATCH),
                dense_update(6, 1, batch_size=3),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [2, 0]}),
                CacheOperation(CacheOperationKind.EXTRACT, {"index": 1}),
            ),
        ),
        OperationSequence(
            "trim-snapshot-restore",
            (
                dense_update(0, 5, batch_size=3, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                dense_update(5, 3, batch_size=3, dtype="float16"),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                dense_update(5, 1, batch_size=3, dtype="float16"),
            ),
        ),
        OperationSequence(
            "merge-and-extend",
            (
                CacheOperation(
                    CacheOperationKind.MERGE,
                    {"rows": [_row(100, 2), _row(200, 5), _row(300, 3)]},
                ),
                CacheOperation(
                    CacheOperationKind.EXTEND,
                    {"rows": [_row(400, 4), _row(500, 1)]},
                ),
                dense_update(600, 1, batch_size=5),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                dense_update(0, 4, batch_size=3),
                CacheOperation(CacheOperationKind.RESET),
                dense_update(20, 2, batch_size=3),
            ),
        ),
    )


def _batch_random_sequences() -> tuple[OperationSequence, ...]:
    sequences = []
    for seed in range(5):
        generator = random.Random(100 + seed)
        cursor = 1000 + seed * 1000
        size = 0
        snapshots: dict[str, int] = {}
        operations = []
        for step in range(25):
            choices = ["update", "update", "reset"]
            if size:
                choices.extend(("trim", "snapshot"))
            if snapshots:
                choices.append("restore")
            action = generator.choice(choices)
            if action == "update":
                count = generator.randint(1, 5)
                operations.append(dense_update(cursor, count, batch_size=3))
                cursor += count
                size += count
            elif action == "trim":
                count = generator.randint(1, min(size, 4))
                operations.append(
                    CacheOperation(CacheOperationKind.TRIM, {"count": count})
                )
                size -= count
            elif action == "snapshot":
                name = f"batch-{step}"
                operations.append(
                    CacheOperation(CacheOperationKind.SNAPSHOT, {"name": name})
                )
                snapshots[name] = size
            elif action == "restore":
                name = generator.choice(sorted(snapshots))
                operations.append(
                    CacheOperation(CacheOperationKind.RESTORE, {"name": name})
                )
                size = snapshots[name]
            else:
                operations.append(CacheOperation(CacheOperationKind.RESET))
                size = 0
        sequences.append(
            OperationSequence(f"batch-state-machine-{seed}", tuple(operations))
        )
    return tuple(sequences)


def _row(start: int, count: int) -> dict[str, Any]:
    operation = dense_update(start, count)
    return dict(operation.payload)


def _random_sequences(
    prefix: str, *, allow_trim: bool
) -> tuple[OperationSequence, ...]:
    sequences: list[OperationSequence] = []
    for seed in range(5):
        generator = random.Random(seed)
        operations: list[CacheOperation] = []
        size = 0
        batch_size = 1
        value_cursor = 1000 + seed * 1000
        snapshots: dict[str, tuple[int, int]] = {}
        for step in range(25):
            choices = ["update", "update", "reset"]
            if size:
                choices.append("snapshot")
            if allow_trim and size:
                choices.append("trim")
            if snapshots:
                choices.append("restore")
            action = generator.choice(choices)
            if action == "update":
                count = generator.randint(1, 7)
                operations.append(
                    dense_update(value_cursor, count, batch_size=batch_size)
                )
                value_cursor += count
                size += count
            elif action == "trim":
                count = generator.randint(1, min(size, 5))
                operations.append(
                    CacheOperation(CacheOperationKind.TRIM, {"count": count})
                )
                size -= count
            elif action == "snapshot":
                name = f"state-{step}"
                operations.append(
                    CacheOperation(CacheOperationKind.SNAPSHOT, {"name": name})
                )
                snapshots[name] = (size, batch_size)
            elif action == "restore":
                name = generator.choice(sorted(snapshots))
                operations.append(
                    CacheOperation(CacheOperationKind.RESTORE, {"name": name})
                )
                size, batch_size = snapshots[name]
            else:
                operations.append(CacheOperation(CacheOperationKind.RESET))
                size = 0
                batch_size = generator.randint(1, 3)
        sequences.append(
            OperationSequence(f"{prefix}-state-machine-{seed}", tuple(operations))
        )
    return tuple(sequences)
