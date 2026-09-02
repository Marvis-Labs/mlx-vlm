from __future__ import annotations

import random
from typing import Any, Callable, Mapping

from ci.kv_cache_contract import (
    CacheCapability,
    CacheCharacteristic,
    CacheContractCase,
    CacheObservation,
    CacheOperation,
    CacheOperationKind,
    OperationSequence,
    StorageProfile,
)
from ci.kv_cache_oracles import ChunkedKVOracle, DenseKVOracle
from ci.kv_cache_profiles.common import MLXDenseCacheAdapter, cache_update


class MLXChunkedCacheAdapter:
    """Expose production chunked-cache lifecycle through semantic observations."""

    capabilities = frozenset(
        {
            CacheCapability.UPDATE,
            CacheCapability.TRIM,
            CacheCapability.RESET,
            CacheCapability.SNAPSHOT_RESTORE,
        }
    )

    def __init__(self, cache_factory: Callable[[], Any]):
        self.cache_factory = cache_factory
        self.cache = cache_factory()
        self.snapshots: dict[str, Any] = {}

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"chunked cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        offset = int(self.cache.offset)
        start_position = int(self.cache.start_position)
        length = offset - start_position
        metadata = {
            "chunk_size": int(self.cache.chunk_size),
            "start_position": start_position,
        }
        keys = getattr(self.cache, "keys", None)
        values = getattr(self.cache, "values", None)
        if keys is None or values is None:
            return CacheObservation(
                logical_keys=(),
                logical_values=(),
                visible_positions=(),
                offset=offset,
                size=0,
                shape=(None, None),
                dtype=(None, None),
                batch_size=0,
                metadata=metadata,
                allocated_bytes=0,
            )
        active_keys = keys[..., :length, :]
        active_values = values[..., :length, :]
        mx.eval(active_keys, active_values)
        return CacheObservation(
            logical_keys=_freeze(active_keys.tolist()),
            logical_values=_freeze(active_values.tolist()),
            visible_positions=tuple(range(start_position, offset)),
            offset=offset,
            size=length,
            shape=(tuple(active_keys.shape), tuple(active_values.shape)),
            dtype=(_dtype_name(active_keys.dtype), _dtype_name(active_values.dtype)),
            batch_size=int(active_keys.shape[0]),
            metadata=metadata,
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        self.cache.maybe_trim_front()
        key_dtype = _mlx_dtype(
            payload.get("key_dtype", payload.get("dtype", "float32"))
        )
        value_dtype = _mlx_dtype(
            payload.get("value_dtype", payload.get("dtype", "float32"))
        )
        keys = mx.array(payload.get("keys"), dtype=key_dtype)
        values = mx.array(payload.get("values"), dtype=value_dtype)
        fetched = self.cache.update_and_fetch(keys, values)
        mx.eval(*fetched)

    def _trim(self, payload: Mapping[str, Any]) -> None:
        self.cache.trim(_non_negative_integer(payload.get("count"), "count"))

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.cache = self.cache_factory()

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        name = _snapshot_name(payload)
        snapshot = _detach(self.cache.prefix_cache_snapshot())
        arrays: list[Any] = []
        _collect_arrays(snapshot, arrays)
        if arrays:
            mx.eval(*arrays)
        self.snapshots[name] = snapshot

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self.cache = self.cache_factory()
        self.cache.prefix_cache_restore(self.snapshots[name])


def segmented_contract_cases() -> tuple[CacheContractCase, ...]:
    from mlx_vlm.models.cache import ChunkedKVCache, ConcatenateKVCache

    shared = frozenset(
        {
            CacheCapability.UPDATE,
            CacheCapability.TRIM,
            CacheCapability.RESET,
            CacheCapability.SNAPSHOT_RESTORE,
        }
    )
    logical_characteristics = frozenset(
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
            name="ChunkedKVCache",
            profile=StorageProfile.SEGMENTED,
            subject_factory=lambda: MLXChunkedCacheAdapter(
                lambda: ChunkedKVCache(chunk_size=8)
            ),
            oracle_factory=lambda: ChunkedKVOracle(chunk_size=8),
            capabilities=shared,
            characteristics=logical_characteristics | {CacheCharacteristic.METADATA},
            sequences=_chunked_sequences() + _chunked_random_sequences(),
        ),
        CacheContractCase(
            name="ConcatenateKVCache",
            profile=StorageProfile.SEGMENTED,
            subject_factory=lambda: MLXDenseCacheAdapter(ConcatenateKVCache, shared),
            oracle_factory=DenseKVOracle,
            capabilities=shared,
            characteristics=logical_characteristics,
            sequences=_concatenate_sequences() + _dense_random_sequences(),
        ),
    )


def _chunked_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "front-eviction",
            (
                cache_update(0, 9),
                cache_update(9, 2),
                cache_update(11, 3),
            ),
        ),
        OperationSequence(
            "snapshot-after-front-eviction",
            (
                cache_update(0, 9, dtype="float16"),
                cache_update(9, 2, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "window"}),
                cache_update(11, 4, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "window"}),
                cache_update(11, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "tail-trim-after-front-eviction",
            (
                cache_update(0, 10),
                cache_update(10, 1),
                CacheOperation(CacheOperationKind.TRIM, {"count": 3}),
                cache_update(8, 2),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                cache_update(0, 12),
                cache_update(12, 1),
                CacheOperation(CacheOperationKind.RESET),
                cache_update(20, 3),
            ),
        ),
    )


def _concatenate_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "append-trim-resume",
            (
                cache_update(0, 3),
                cache_update(3, 4),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                cache_update(5, 2),
            ),
        ),
        OperationSequence(
            "snapshot-restore-resume",
            (
                cache_update(0, 4, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                cache_update(4, 3, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                cache_update(4, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                cache_update(0, 3),
                CacheOperation(CacheOperationKind.RESET),
                cache_update(10, 2),
            ),
        ),
    )


def _chunked_random_sequences() -> tuple[OperationSequence, ...]:
    return tuple(
        _random_sequence(f"chunked-state-machine-{seed}", seed, chunk_size=8)
        for seed in range(5)
    )


def _dense_random_sequences() -> tuple[OperationSequence, ...]:
    return tuple(
        _random_sequence(f"concatenate-state-machine-{seed}", seed) for seed in range(5)
    )


def _random_sequence(
    name: str, seed: int, chunk_size: int | None = None
) -> OperationSequence:
    generator = random.Random(seed)
    operations: list[CacheOperation] = []
    size = 0
    cursor = 1000 + seed * 1000
    snapshots: dict[str, int] = {}
    for step in range(25):
        choices = ["update", "update", "reset"]
        if size:
            choices.extend(("trim", "snapshot"))
        if snapshots:
            choices.append("restore")
        action = generator.choice(choices)
        if action == "update":
            if chunk_size is not None and size > chunk_size:
                size = chunk_size
            count = generator.randint(1, 5)
            operations.append(cache_update(cursor, count))
            cursor += count
            size += count
        elif action == "trim":
            count = generator.randint(1, min(size, 4))
            operations.append(CacheOperation(CacheOperationKind.TRIM, {"count": count}))
            size -= count
        elif action == "snapshot":
            snapshot = f"state-{step}"
            operations.append(
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": snapshot})
            )
            snapshots[snapshot] = size
        elif action == "restore":
            snapshot = generator.choice(sorted(snapshots))
            operations.append(
                CacheOperation(CacheOperationKind.RESTORE, {"name": snapshot})
            )
            size = snapshots[snapshot]
        else:
            operations.append(CacheOperation(CacheOperationKind.RESET))
            size = 0
    return OperationSequence(name, tuple(operations))


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _mlx_dtype(value: Any) -> Any:
    import mlx.core as mx

    if not isinstance(value, str) or not hasattr(mx, value):
        raise ValueError(f"unsupported MLX dtype: {value}")
    return getattr(mx, value)


def _detach(value: Any) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return mx.contiguous(mx.array(value, dtype=value.dtype))
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


def _collect_arrays(value: Any, output: list[Any]) -> None:
    import mlx.core as mx

    if isinstance(value, mx.array):
        output.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_arrays(item, output)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_arrays(item, output)


def _snapshot_name(payload: Mapping[str, Any]) -> str:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("snapshot name must be a non-empty string")
    return name


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
