from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ci.kv_cache_contract import (
    CacheCapability,
    CacheObservation,
    CacheOperation,
    CacheOperationKind,
)


@dataclass(frozen=True)
class _DenseState:
    keys: tuple[Any, ...]
    values: tuple[Any, ...]
    positions: tuple[int, ...]
    key_layout: tuple[int, int, int] | None
    value_layout: tuple[int, int, int] | None
    key_dtype: str | None
    value_dtype: str | None


class DenseKVOracle:
    """Reference dense cache implemented without MLX cache machinery."""

    capabilities = frozenset(
        {
            CacheCapability.UPDATE,
            CacheCapability.TRIM,
            CacheCapability.RESET,
            CacheCapability.SNAPSHOT_RESTORE,
            CacheCapability.EXTRACT,
            CacheCapability.FILTER,
        }
    )

    def __init__(self):
        self._keys: tuple[Any, ...] = ()
        self._values: tuple[Any, ...] = ()
        self._positions: tuple[int, ...] = ()
        self._key_layout: tuple[int, int, int] | None = None
        self._value_layout: tuple[int, int, int] | None = None
        self._key_dtype: str | None = None
        self._value_dtype: str | None = None
        self._snapshots: dict[str, _DenseState] = {}

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
            raise ValueError(f"dense oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        key_shape = _logical_shape(self._key_layout, len(self._positions))
        value_shape = _logical_shape(self._value_layout, len(self._positions))
        return CacheObservation(
            logical_keys=self._keys,
            logical_values=self._values,
            visible_positions=self._positions,
            offset=len(self._positions),
            size=len(self._positions),
            shape=(key_shape, value_shape),
            dtype=(self._key_dtype, self._value_dtype),
            batch_size=key_shape[0] if key_shape is not None else 0,
            metadata={},
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        keys = _freeze(payload.get("keys"))
        values = _freeze(payload.get("values"))
        key_shape = _require_rank_four(keys, "keys")
        value_shape = _require_rank_four(values, "values")
        if key_shape[:3] != value_shape[:3]:
            raise ValueError("keys and values must share batch, head, and time axes")
        key_dtype = str(payload.get("key_dtype", payload.get("dtype", "float32")))
        value_dtype = str(payload.get("value_dtype", payload.get("dtype", "float32")))
        if self._key_layout is not None and self._value_layout is not None:
            if self._key_layout != (key_shape[0], key_shape[1], key_shape[3]):
                raise ValueError("key update shape is incompatible with cached keys")
            if self._value_layout != (
                value_shape[0],
                value_shape[1],
                value_shape[3],
            ):
                raise ValueError(
                    "value update shape is incompatible with cached values"
                )
            if (self._key_dtype, self._value_dtype) != (key_dtype, value_dtype):
                raise ValueError("update dtype differs from cached dtype")
            self._keys = _concatenate_time(self._keys, keys)
            self._values = _concatenate_time(self._values, values)
        else:
            self._keys = keys
            self._values = values
            self._key_layout = (key_shape[0], key_shape[1], key_shape[3])
            self._value_layout = (value_shape[0], value_shape[1], value_shape[3])
            self._key_dtype = key_dtype
            self._value_dtype = value_dtype
        start = len(self._positions)
        self._positions += tuple(range(start, start + key_shape[2]))

    def _trim(self, payload: Mapping[str, Any]) -> None:
        count = _positive_or_zero_integer(payload.get("count"), "count")
        if count > len(self._positions):
            raise ValueError("cannot trim more tokens than the cache contains")
        if count == 0:
            return
        self._keys = _slice_time(self._keys, len(self._positions) - count)
        self._values = _slice_time(self._values, len(self._positions) - count)
        self._positions = self._positions[:-count]

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self._keys = ()
        self._values = ()
        self._positions = ()
        self._key_layout = None
        self._value_layout = None
        self._key_dtype = None
        self._value_dtype = None

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self._snapshots[name] = _DenseState(
            keys=self._keys,
            values=self._values,
            positions=self._positions,
            key_layout=self._key_layout,
            value_layout=self._value_layout,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
        )

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self._snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        state = self._snapshots[name]
        self._keys = state.keys
        self._values = state.values
        self._positions = state.positions
        self._key_layout = state.key_layout
        self._value_layout = state.value_layout
        self._key_dtype = state.key_dtype
        self._value_dtype = state.value_dtype

    def _extract(self, payload: Mapping[str, Any]) -> None:
        index = _integer(payload.get("index"), "index")
        batch_size = _shape(self._keys)[0] if self._keys else 0
        index = _normalized_index(index, batch_size)
        self._keys = (self._keys[index],)
        self._values = (self._values[index],)
        if self._key_layout is not None:
            self._key_layout = (1, self._key_layout[1], self._key_layout[2])
        if self._value_layout is not None:
            self._value_layout = (1, self._value_layout[1], self._value_layout[2])

    def _filter(self, payload: Mapping[str, Any]) -> None:
        indices = payload.get("indices")
        if not _sequence(indices):
            raise ValueError("indices must be a sequence")
        batch_size = _shape(self._keys)[0] if self._keys else 0
        normalized = tuple(
            _normalized_index(_integer(index, "index"), batch_size) for index in indices
        )
        self._keys = tuple(self._keys[index] for index in normalized)
        self._values = tuple(self._values[index] for index in normalized)
        if self._key_layout is not None:
            self._key_layout = (
                len(normalized),
                self._key_layout[1],
                self._key_layout[2],
            )
        if self._value_layout is not None:
            self._value_layout = (
                len(normalized),
                self._value_layout[1],
                self._value_layout[2],
            )


class WindowedKVOracle:
    """Reference sliding-window cache implemented without MLX cache machinery."""

    capabilities = frozenset({CacheCapability.UPDATE, CacheCapability.RESET})

    def __init__(self, max_size: int, keep: int = 0):
        if max_size <= 0 or keep < 0 or keep >= max_size:
            raise ValueError("invalid window configuration")
        self.max_size = max_size
        self.keep = keep
        self._keys: tuple[Any, ...] = ()
        self._values: tuple[Any, ...] = ()
        self._positions: tuple[int, ...] = ()
        self._key_layout: tuple[int, int, int] | None = None
        self._value_layout: tuple[int, int, int] | None = None
        self._key_dtype: str | None = None
        self._value_dtype: str | None = None
        self._offset = 0

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(f"windowed oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        key_shape = _logical_shape(self._key_layout, len(self._positions))
        value_shape = _logical_shape(self._value_layout, len(self._positions))
        return CacheObservation(
            logical_keys=self._keys,
            logical_values=self._values,
            visible_positions=self._positions,
            offset=self._offset,
            size=len(self._positions),
            shape=(key_shape, value_shape),
            dtype=(self._key_dtype, self._value_dtype),
            batch_size=key_shape[0] if key_shape is not None else 0,
            metadata={"max_size": self.max_size, "keep": self.keep},
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        keys = _freeze(payload.get("keys"))
        values = _freeze(payload.get("values"))
        key_shape = _require_rank_four(keys, "keys")
        value_shape = _require_rank_four(values, "values")
        if key_shape[:3] != value_shape[:3]:
            raise ValueError("keys and values must share batch, head, and time axes")
        key_dtype = str(payload.get("key_dtype", payload.get("dtype", "float32")))
        value_dtype = str(payload.get("value_dtype", payload.get("dtype", "float32")))
        if self._key_layout is not None and self._value_layout is not None:
            if self._key_layout != (key_shape[0], key_shape[1], key_shape[3]):
                raise ValueError("key update shape is incompatible with cached keys")
            if self._value_layout != (
                value_shape[0],
                value_shape[1],
                value_shape[3],
            ):
                raise ValueError(
                    "value update shape is incompatible with cached values"
                )
            if (self._key_dtype, self._value_dtype) != (key_dtype, value_dtype):
                raise ValueError("update dtype differs from cached dtype")
            self._keys = _concatenate_time(self._keys, keys)
            self._values = _concatenate_time(self._values, values)
        else:
            self._keys = keys
            self._values = values
            self._key_layout = (key_shape[0], key_shape[1], key_shape[3])
            self._value_layout = (value_shape[0], value_shape[1], value_shape[3])
            self._key_dtype = key_dtype
            self._value_dtype = value_dtype
        self._positions += tuple(range(self._offset, self._offset + key_shape[2]))
        self._offset += key_shape[2]
        self._keys = _window_time(self._keys, self.max_size, self.keep)
        self._values = _window_time(self._values, self.max_size, self.keep)
        self._positions = _window_positions(self._positions, self.max_size, self.keep)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self._keys = ()
        self._values = ()
        self._positions = ()
        self._key_layout = None
        self._value_layout = None
        self._key_dtype = None
        self._value_dtype = None
        self._offset = 0


@dataclass(frozen=True)
class _ChunkedState:
    keys: tuple[Any, ...]
    values: tuple[Any, ...]
    positions: tuple[int, ...]
    key_layout: tuple[int, int, int] | None
    value_layout: tuple[int, int, int] | None
    key_dtype: str | None
    value_dtype: str | None
    offset: int
    start_position: int


class ChunkedKVOracle:
    """Reference chunked cache modeled as logical tokens and absolute positions."""

    capabilities = frozenset(
        {
            CacheCapability.UPDATE,
            CacheCapability.TRIM,
            CacheCapability.RESET,
            CacheCapability.SNAPSHOT_RESTORE,
        }
    )

    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._keys: tuple[Any, ...] = ()
        self._values: tuple[Any, ...] = ()
        self._positions: tuple[int, ...] = ()
        self._key_layout: tuple[int, int, int] | None = None
        self._value_layout: tuple[int, int, int] | None = None
        self._key_dtype: str | None = None
        self._value_dtype: str | None = None
        self._offset = 0
        self._start_position = 0
        self._snapshots: dict[str, _ChunkedState] = {}

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
            raise ValueError(f"chunked oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        key_shape = _logical_shape(self._key_layout, len(self._positions))
        value_shape = _logical_shape(self._value_layout, len(self._positions))
        return CacheObservation(
            logical_keys=self._keys,
            logical_values=self._values,
            visible_positions=self._positions,
            offset=self._offset,
            size=len(self._positions),
            shape=(key_shape, value_shape),
            dtype=(self._key_dtype, self._value_dtype),
            batch_size=key_shape[0] if key_shape is not None else 0,
            metadata={
                "chunk_size": self.chunk_size,
                "start_position": self._start_position,
            },
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        self._trim_front()
        keys = _freeze(payload.get("keys"))
        values = _freeze(payload.get("values"))
        key_shape = _require_rank_four(keys, "keys")
        value_shape = _require_rank_four(values, "values")
        if key_shape[:3] != value_shape[:3]:
            raise ValueError("keys and values must share batch, head, and time axes")
        key_dtype = str(payload.get("key_dtype", payload.get("dtype", "float32")))
        value_dtype = str(payload.get("value_dtype", payload.get("dtype", "float32")))
        if self._key_layout is not None and self._value_layout is not None:
            if self._key_layout != (key_shape[0], key_shape[1], key_shape[3]):
                raise ValueError("key update shape is incompatible with cached keys")
            if self._value_layout != (
                value_shape[0],
                value_shape[1],
                value_shape[3],
            ):
                raise ValueError(
                    "value update shape is incompatible with cached values"
                )
            if (self._key_dtype, self._value_dtype) != (key_dtype, value_dtype):
                raise ValueError("update dtype differs from cached dtype")
            self._keys = _concatenate_time(self._keys, keys)
            self._values = _concatenate_time(self._values, values)
        else:
            self._keys = keys
            self._values = values
            self._key_layout = (key_shape[0], key_shape[1], key_shape[3])
            self._value_layout = (value_shape[0], value_shape[1], value_shape[3])
            self._key_dtype = key_dtype
            self._value_dtype = value_dtype
        self._positions += tuple(range(self._offset, self._offset + key_shape[2]))
        self._offset += key_shape[2]

    def _trim_front(self) -> None:
        excess = len(self._positions) - self.chunk_size
        if excess <= 0:
            return
        self._keys = _drop_time(self._keys, excess)
        self._values = _drop_time(self._values, excess)
        self._positions = self._positions[excess:]
        self._start_position += excess

    def _trim(self, payload: Mapping[str, Any]) -> None:
        count = min(
            len(self._positions),
            _positive_or_zero_integer(payload.get("count"), "count"),
        )
        if count == 0:
            return
        self._keys = _slice_time(self._keys, len(self._positions) - count)
        self._values = _slice_time(self._values, len(self._positions) - count)
        self._positions = self._positions[:-count]
        self._offset -= count

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self._keys = ()
        self._values = ()
        self._positions = ()
        self._key_layout = None
        self._value_layout = None
        self._key_dtype = None
        self._value_dtype = None
        self._offset = 0
        self._start_position = 0

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        self._snapshots[_snapshot_name(payload)] = _ChunkedState(
            keys=self._keys,
            values=self._values,
            positions=self._positions,
            key_layout=self._key_layout,
            value_layout=self._value_layout,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            offset=self._offset,
            start_position=self._start_position,
        )

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self._snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        state = self._snapshots[name]
        self._keys = state.keys
        self._values = state.values
        self._positions = state.positions
        self._key_layout = state.key_layout
        self._value_layout = state.value_layout
        self._key_dtype = state.key_dtype
        self._value_dtype = state.value_dtype
        self._offset = state.offset
        self._start_position = state.start_position


def _freeze(value: Any) -> tuple[Any, ...]:
    if not _sequence(value):
        raise ValueError("cache values must be nested sequences")
    return tuple(_freeze(item) if _sequence(item) else item for item in value)


def _shape(value: tuple[Any, ...]) -> tuple[int, ...]:
    if not value:
        return (0,)
    child_shapes = {_shape(item) if isinstance(item, tuple) else () for item in value}
    if len(child_shapes) != 1:
        raise ValueError("cache values must be rectangular")
    return (len(value),) + child_shapes.pop()


def _logical_shape(
    layout: tuple[int, int, int] | None, length: int
) -> tuple[int, int, int, int] | None:
    if layout is None:
        return None
    batch, heads, channels = layout
    return batch, heads, length, channels


def _require_rank_four(value: tuple[Any, ...], name: str) -> tuple[int, int, int, int]:
    shape = _shape(value)
    if len(shape) != 4:
        raise ValueError(f"{name} must have shape [batch, heads, time, channels]")
    return shape


def _concatenate_time(
    current: tuple[Any, ...], update: tuple[Any, ...]
) -> tuple[Any, ...]:
    return tuple(
        tuple(
            current_head + update_head for current_head, update_head in zip(left, right)
        )
        for left, right in zip(current, update)
    )


def _slice_time(values: tuple[Any, ...], stop: int) -> tuple[Any, ...]:
    return tuple(tuple(head[:stop] for head in batch) for batch in values)


def _drop_time(values: tuple[Any, ...], start: int) -> tuple[Any, ...]:
    return tuple(tuple(head[start:] for head in batch) for batch in values)


def _window_time(values: tuple[Any, ...], max_size: int, keep: int) -> tuple[Any, ...]:
    length = _shape(values)[2]
    if length <= max_size:
        return values
    recent = max_size - keep
    return tuple(
        tuple(head[:keep] + head[-recent:] for head in batch) for batch in values
    )


def _window_positions(
    positions: tuple[int, ...], max_size: int, keep: int
) -> tuple[int, ...]:
    if len(positions) <= max_size:
        return positions
    return positions[:keep] + positions[-(max_size - keep) :]


def _snapshot_name(payload: Mapping[str, Any]) -> str:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("snapshot name must be a non-empty string")
    return name


def _positive_or_zero_integer(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return parsed


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _normalized_index(index: int, size: int) -> int:
    normalized = index + size if index < 0 else index
    if normalized < 0 or normalized >= size:
        raise IndexError(f"batch index {index} is out of range for size {size}")
    return normalized


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )
