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
    Tolerance,
)
from ci.kv_cache_oracles import DenseKVOracle

SINGLE_CAPABILITIES = frozenset(
    {
        CacheCapability.UPDATE,
        CacheCapability.TRIM,
        CacheCapability.RESET,
        CacheCapability.SNAPSHOT_RESTORE,
    }
)
BATCH_CAPABILITIES = frozenset(
    {
        CacheCapability.UPDATE,
        CacheCapability.TRIM,
        CacheCapability.RESET,
        CacheCapability.EXTRACT,
        CacheCapability.FILTER,
    }
)


class MLXQuantizedCacheAdapter:
    """Observe quantized KV storage through its dequantized logical content."""

    def __init__(
        self,
        cache_factory: Callable[[], Any],
        capabilities: frozenset[CacheCapability],
    ):
        self.cache_factory = cache_factory
        self.capabilities = capabilities
        self.cache = cache_factory()
        self.snapshots: dict[str, Any] = {}
        self.layout: tuple[int, int, int, int, str, str] | None = None

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.EXTRACT: self._extract,
            CacheOperationKind.FILTER: self._filter,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"quantized cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        length = _length(self.cache)
        keys, values = self.cache.dequantize_for_apc()
        if keys is None or values is None:
            if self.layout is not None and not self.cache.empty():
                batch, heads, key_channels, value_channels, key_dtype, value_dtype = (
                    self.layout
                )
                empty_keys = _empty_values(batch, heads, key_channels)
                empty_values = _empty_values(batch, heads, value_channels)
                return CacheObservation(
                    logical_keys=empty_keys,
                    logical_values=empty_values,
                    visible_positions=(),
                    offset=0,
                    size=0,
                    shape=(
                        (batch, heads, 0, key_channels),
                        (batch, heads, 0, value_channels),
                    ),
                    dtype=(key_dtype, value_dtype),
                    batch_size=batch,
                    metadata={},
                    allocated_bytes=int(self.cache.nbytes),
                )
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
        mx.eval(keys, values)
        active_keys = keys[..., :length, :]
        active_values = values[..., :length, :]
        return CacheObservation(
            logical_keys=_freeze(active_keys.tolist()),
            logical_values=_freeze(active_values.tolist()),
            visible_positions=tuple(range(length)),
            offset=length,
            size=length,
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
        self.layout = (
            int(keys.shape[0]),
            int(keys.shape[1]),
            int(keys.shape[3]),
            int(values.shape[3]),
            _dtype_name(keys.dtype),
            _dtype_name(values.dtype),
        )
        fetched = self.cache.update_and_fetch(keys, values)
        mx.eval(*_flatten_arrays(fetched))

    def _trim(self, payload: Mapping[str, Any]) -> None:
        self.cache.trim(_non_negative_integer(payload.get("count"), "count"))

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.cache = self.cache_factory()
        self.layout = None

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self.snapshots[name] = _detach(self.cache.prefix_cache_snapshot())

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self.cache = self.cache_factory()
        self.cache.prefix_cache_restore(self.snapshots[name])

    def _extract(self, payload: Mapping[str, Any]) -> None:
        self.cache = self.cache.extract(_integer(payload.get("index"), "index"))
        if self.layout is not None:
            self.layout = (1, *self.layout[1:])

    def _filter(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        indices = payload.get("indices")
        self.cache.filter(mx.array(indices, dtype=mx.int32))
        if self.layout is not None:
            self.layout = (len(indices), *self.layout[1:])


def quantized_contract_cases() -> tuple[CacheContractCase, ...]:
    from mlx_vlm.models.cache import BatchQuantizedKVCache, QuantizedKVCache

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
    tolerance = Tolerance(absolute=0.05, relative=0.02)
    return (
        CacheContractCase(
            name="QuantizedKVCache",
            profile=StorageProfile.QUANTIZED,
            subject_factory=lambda: MLXQuantizedCacheAdapter(
                lambda: QuantizedKVCache(group_size=32, bits=8),
                SINGLE_CAPABILITIES,
            ),
            oracle_factory=DenseKVOracle,
            capabilities=SINGLE_CAPABILITIES,
            characteristics=characteristics,
            sequences=_single_sequences() + _random_sequences("single", batch_size=1),
            tolerance=tolerance,
        ),
        CacheContractCase(
            name="BatchQuantizedKVCache",
            profile=StorageProfile.QUANTIZED,
            subject_factory=lambda: MLXQuantizedCacheAdapter(
                lambda: BatchQuantizedKVCache([0, 0, 0], group_size=32, bits=8),
                BATCH_CAPABILITIES,
            ),
            oracle_factory=DenseKVOracle,
            capabilities=BATCH_CAPABILITIES,
            characteristics=characteristics,
            sequences=_batch_sequences()
            + _random_sequences("batch", batch_size=3, allow_extract=False),
            tolerance=tolerance,
        ),
    )


def _single_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "incremental-append",
            (_update(0, 3), _update(3, 5), _update(8, 2)),
        ),
        OperationSequence(
            "trim-and-resume",
            (
                _update(0, 7),
                CacheOperation(CacheOperationKind.TRIM, {"count": 3}),
                _update(4, 2),
            ),
        ),
        OperationSequence(
            "snapshot-restore",
            (
                _update(0, 5, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                _update(5, 3, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                _update(5, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "allocation-boundary",
            (_update(0, 255), _update(255, 2)),
        ),
    )


def _batch_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "batch-filter-extract",
            (
                _update(0, 5, batch_size=3),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [2, 0]}),
                CacheOperation(CacheOperationKind.EXTRACT, {"index": 1}),
            ),
        ),
        OperationSequence(
            "batch-trim-resume",
            (
                _update(0, 6, batch_size=3),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                _update(4, 3, batch_size=3),
            ),
        ),
        OperationSequence(
            "batch-reset-reuse",
            (
                _update(0, 4, batch_size=3),
                CacheOperation(CacheOperationKind.RESET),
                _update(20, 2, batch_size=3),
            ),
        ),
    )


def _random_sequences(
    prefix: str, batch_size: int, allow_extract: bool = False
) -> tuple[OperationSequence, ...]:
    sequences: list[OperationSequence] = []
    for seed in range(4):
        generator = random.Random(seed)
        operations: list[CacheOperation] = []
        size = 0
        cursor = 1000 + seed * 1000
        current_batch = batch_size
        for _ in range(20):
            choices = ["update", "update", "reset"]
            if size:
                choices.append("trim")
            if allow_extract and size and current_batch > 1:
                choices.append("extract")
            action = generator.choice(choices)
            if action == "update":
                count = generator.randint(1, 6)
                operations.append(_update(cursor, count, batch_size=current_batch))
                cursor += count
                size += count
            elif action == "trim":
                count = generator.randint(1, min(size, 4))
                operations.append(
                    CacheOperation(CacheOperationKind.TRIM, {"count": count})
                )
                size -= count
            elif action == "extract":
                operations.append(
                    CacheOperation(CacheOperationKind.EXTRACT, {"index": 0})
                )
                current_batch = 1
            else:
                operations.append(CacheOperation(CacheOperationKind.RESET))
                size = 0
                current_batch = batch_size
        sequences.append(
            OperationSequence(f"{prefix}-state-machine-{seed}", tuple(operations))
        )
    return tuple(sequences)


def _update(
    start: int,
    count: int,
    batch_size: int = 1,
    dtype: str = "float32",
) -> CacheOperation:
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {
            "keys": _values(start, count, batch_size, family=0),
            "values": _values(start, count, batch_size, family=1),
            "dtype": dtype,
        },
    )


def _values(
    start: int, count: int, batch_size: int, family: int
) -> list[list[list[list[float]]]]:
    return [
        [
            [
                [
                    float(
                        (
                            family * 13
                            + batch * 11
                            + head * 7
                            + (start + position) * 5
                            + channel * 3
                        )
                        % 31
                        - 15
                    )
                    / 8.0
                    for channel in range(32)
                ]
                for position in range(count)
            ]
            for head in range(2)
        ]
        for batch in range(batch_size)
    ]


def _length(cache: Any) -> int:
    return int(cache._idx) if hasattr(cache, "_idx") else int(cache.offset)


def _empty_values(batch: int, heads: int, channels: int) -> tuple[Any, ...]:
    return tuple(tuple(() for _ in range(heads)) for _ in range(batch))


def _flatten_arrays(value: Any) -> list[Any]:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    arrays: list[Any] = []
    for item in value:
        arrays.extend(_flatten_arrays(item))
    return arrays


def _detach(value: Any) -> Any:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return mx.contiguous(mx.array(value, dtype=value.dtype))
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


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
