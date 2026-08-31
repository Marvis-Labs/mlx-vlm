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
from ci.kv_cache_oracles import DenseKVOracle


class MLXDenseCacheAdapter:
    """Expose an MLX dense cache through normalized contract observations."""

    def __init__(
        self,
        cache_factory: Callable[[], Any],
        capabilities: frozenset[CacheCapability],
    ):
        self.cache_factory = cache_factory
        self.capabilities = capabilities
        self.cache = cache_factory()
        self.snapshots: dict[str, Any] = {}

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.EXTRACT: self._extract,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"dense cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        offset = _offset(self.cache)
        keys = getattr(self.cache, "keys", None)
        values = getattr(self.cache, "values", None)
        if keys is None or values is None:
            return CacheObservation(
                logical_keys=(),
                logical_values=(),
                visible_positions=(),
                offset=0,
                size=0,
                shape=(None, None),
                dtype=(None, None),
                batch_size=0,
                metadata={},
                allocated_bytes=0,
            )
        active_keys = keys[..., :offset, :]
        active_values = values[..., :offset, :]
        mx.eval(active_keys, active_values)
        return CacheObservation(
            logical_keys=_freeze(active_keys.tolist()),
            logical_values=_freeze(active_values.tolist()),
            visible_positions=tuple(range(offset)),
            offset=offset,
            size=offset,
            shape=(tuple(active_keys.shape), tuple(active_values.shape)),
            dtype=(_dtype_name(active_keys.dtype), _dtype_name(active_values.dtype)),
            batch_size=int(active_keys.shape[0]),
            metadata={},
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

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
        snapshot = self.cache.prefix_cache_snapshot()
        detached = _detach(snapshot)
        arrays: list[Any] = []
        _collect_arrays(detached, arrays)
        if arrays:
            mx.eval(*arrays)
        self.snapshots[name] = detached

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self.cache = self.cache_factory()
        self.cache.prefix_cache_restore(self.snapshots[name])

    def _extract(self, payload: Mapping[str, Any]) -> None:
        self.cache = self.cache.extract(_integer(payload.get("index"), "index"))


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


def dense_update(
    start: int,
    count: int,
    *,
    batch_size: int = 1,
    heads: int = 2,
    key_channels: int = 2,
    value_channels: int = 3,
    dtype: str = "float32",
) -> CacheOperation:
    keys = _tensor_values(
        start,
        count,
        batch_size=batch_size,
        heads=heads,
        channels=key_channels,
        family=0,
    )
    values = _tensor_values(
        start,
        count,
        batch_size=batch_size,
        heads=heads,
        channels=value_channels,
        family=1,
    )
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {"keys": keys, "values": values, "dtype": dtype},
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


def _tensor_values(
    start: int,
    count: int,
    *,
    batch_size: int,
    heads: int,
    channels: int,
    family: int,
) -> list[list[list[list[float]]]]:
    return [
        [
            [
                [
                    float(
                        family * 1000
                        + batch * 200
                        + head * 100
                        + (start + position) * 2
                        + channel
                    )
                    for channel in range(channels)
                ]
                for position in range(count)
            ]
            for head in range(heads)
        ]
        for batch in range(batch_size)
    ]


def _offset(cache: Any) -> int:
    if hasattr(cache, "cache_length"):
        return int(cache.cache_length)
    return int(cache.offset)


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
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return parsed


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
