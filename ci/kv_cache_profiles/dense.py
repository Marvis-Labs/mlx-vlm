from __future__ import annotations

import random

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
    from mlx_vlm.models.cache import KVCache, SimpleKVCache

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
