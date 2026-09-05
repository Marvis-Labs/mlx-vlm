from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from ci.kv_cache_contract import (
    CacheCapability,
    CacheObservation,
    CacheOperation,
    CacheOperationKind,
    StorageProfile,
)

BATCH_CAPABILITIES = frozenset(
    {
        CacheCapability.UPDATE,
        CacheCapability.TRIM,
        CacheCapability.RESET,
        CacheCapability.SNAPSHOT_RESTORE,
        CacheCapability.EXTRACT,
        CacheCapability.MERGE,
        CacheCapability.EXTEND,
        CacheCapability.FILTER,
        CacheCapability.BATCH_LIFECYCLE,
    }
)


class BatchKVOracle:
    """Model variable-length KV batches without MLX cache machinery."""

    capabilities = BATCH_CAPABILITIES

    def __init__(
        self,
        batch_size: int,
        *,
        profile: StorageProfile,
        max_size: int | None = None,
    ):
        self.profile = profile
        self.max_size = max_size
        self._reset_rows(batch_size)
        self.snapshots: dict[str, Any] = {}

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.EXTRACT: self._extract,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.EXTEND: self._extend,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(f"batch oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        key_shapes = tuple(_row_shape(row, self.key_layout) for row in self.keys)
        value_shapes = tuple(_row_shape(row, self.value_layout) for row in self.values)
        positions = tuple(
            tuple(range(offset - _time_size(row), offset))
            for row, offset in zip(self.keys, self.offsets)
        )
        return CacheObservation(
            logical_keys=tuple(self.keys),
            logical_values=tuple(self.values),
            visible_positions=positions,
            offset=tuple(self.offsets),
            size=tuple(_time_size(row) for row in self.keys),
            shape=(key_shapes, value_shapes),
            dtype=(self.key_dtype, self.value_dtype),
            batch_size=len(self.keys),
            metadata={
                "profile": self.profile.value,
                "max_size": self.max_size,
            },
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        keys = _freeze(payload.get("keys"))
        values = _freeze(payload.get("values"))
        if len(keys) != len(self.keys) or len(values) != len(self.values):
            raise ValueError("update batch size differs from cache batch size")
        key_dtype = str(payload.get("key_dtype", payload.get("dtype", "float32")))
        value_dtype = str(payload.get("value_dtype", payload.get("dtype", "float32")))
        if self.key_dtype is not None and (self.key_dtype, self.value_dtype) != (
            key_dtype,
            value_dtype,
        ):
            raise ValueError("update dtype differs from cached dtype")
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.key_layout = _row_layout(keys)
        self.value_layout = _row_layout(values)
        for row in range(len(self.keys)):
            key_row = keys[row]
            value_row = values[row]
            count = _time_size(key_row)
            consumed = min(self.pending_left[row], count)
            self.pending_left[row] -= consumed
            key_row = _slice_time_range(key_row, consumed, count)
            value_row = _slice_time_range(value_row, consumed, count)
            self.keys[row] = _concatenate_time(self.keys[row], key_row)
            self.values[row] = _concatenate_time(self.values[row], value_row)
            self.offsets[row] += count - consumed
        if not any(self.pending_right):
            self._apply_windows()

    def _trim(self, payload: Mapping[str, Any]) -> None:
        count = _non_negative_integer(payload.get("count"), "count")
        for row in range(len(self.keys)):
            removed = min(count, _time_size(self.keys[row]))
            self.keys[row] = _slice_time(self.keys[row], -removed)
            self.values[row] = _slice_time(self.values[row], -removed)
            self.offsets[row] -= removed

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self._reset_rows(len(self.keys))

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshots[_snapshot_name(payload)] = deepcopy(self._state())

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self._load_state(deepcopy(self.snapshots[name]))

    def _extract(self, payload: Mapping[str, Any]) -> None:
        index = _normalized_index(payload.get("index"), len(self.keys))
        for name in ("keys", "values", "offsets", "pending_left", "pending_right"):
            values = getattr(self, name)
            setattr(self, name, [values[index]])

    def _merge(self, payload: Mapping[str, Any]) -> None:
        rows = _rows(payload)
        self._load_rows(rows)

    def _extend(self, payload: Mapping[str, Any]) -> None:
        rows = _rows(payload)
        self.keys.extend(row["keys"] for row in rows)
        self.values.extend(row["values"] for row in rows)
        self.offsets.extend(_time_size(row["keys"]) for row in rows)
        self.pending_left.extend(0 for _ in rows)
        self.pending_right.extend(0 for _ in rows)
        self._set_row_dtypes(rows)
        self._apply_windows()

    def _filter(self, payload: Mapping[str, Any]) -> None:
        indices = [
            _normalized_index(index, len(self.keys))
            for index in payload.get("indices", [])
        ]
        for name in ("keys", "values", "offsets", "pending_left", "pending_right"):
            values = getattr(self, name)
            setattr(self, name, [values[index] for index in indices])

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        left_padding = payload.get("left_padding")
        if left_padding is not None:
            _require_batch_values(left_padding, len(self.keys), "left_padding")
            self.pending_left = [
                current + int(value)
                for current, value in zip(self.pending_left, left_padding)
            ]
        right_padding = payload.get("right_padding")
        if right_padding is not None:
            _require_batch_values(right_padding, len(self.keys), "right_padding")
            self.pending_right = [int(value) for value in right_padding]

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        for row, count in enumerate(self.pending_right):
            if count:
                self.keys[row] = _slice_time(self.keys[row], -count)
                self.values[row] = _slice_time(self.values[row], -count)
                self.offsets[row] -= count
        self.pending_right = [0] * len(self.keys)
        self._apply_windows()

    def _apply_windows(self) -> None:
        if self.max_size is None:
            return
        for row in range(len(self.keys)):
            length = _time_size(self.keys[row])
            if length > self.max_size:
                start = length - self.max_size
                self.keys[row] = _slice_time_range(self.keys[row], start, length)
                self.values[row] = _slice_time_range(self.values[row], start, length)

    def _load_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.keys = [row["keys"] for row in rows]
        self.values = [row["values"] for row in rows]
        self.offsets = [_time_size(row["keys"]) for row in rows]
        self.pending_left = [0] * len(rows)
        self.pending_right = [0] * len(rows)
        self._set_row_dtypes(rows)
        self._apply_windows()

    def _set_row_dtypes(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        key_dtype = str(rows[0].get("key_dtype", rows[0].get("dtype", "float32")))
        value_dtype = str(rows[0].get("value_dtype", rows[0].get("dtype", "float32")))
        if self.key_dtype is not None and (self.key_dtype, self.value_dtype) != (
            key_dtype,
            value_dtype,
        ):
            raise ValueError("row dtype differs from cached dtype")
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.key_layout = _row_layout(tuple(row["keys"] for row in rows))
        self.value_layout = _row_layout(tuple(row["values"] for row in rows))

    def _reset_rows(self, batch_size: int) -> None:
        self.keys = [() for _ in range(batch_size)]
        self.values = [() for _ in range(batch_size)]
        self.offsets = [0 for _ in range(batch_size)]
        self.pending_left = [0 for _ in range(batch_size)]
        self.pending_right = [0 for _ in range(batch_size)]
        self.key_dtype: str | None = None
        self.value_dtype: str | None = None
        self.key_layout: tuple[int, int] | None = None
        self.value_layout: tuple[int, int] | None = None

    def _state(self) -> tuple[Any, ...]:
        return (
            self.keys,
            self.values,
            self.offsets,
            self.pending_left,
            self.pending_right,
            self.key_dtype,
            self.value_dtype,
            self.key_layout,
            self.value_layout,
        )

    def _load_state(self, state: tuple[Any, ...]) -> None:
        (
            self.keys,
            self.values,
            self.offsets,
            self.pending_left,
            self.pending_right,
            self.key_dtype,
            self.value_dtype,
            self.key_layout,
            self.value_layout,
        ) = state


class MLXBatchKVAdapter:
    """Normalize production batch caches into row-wise semantic observations."""

    capabilities = BATCH_CAPABILITIES

    def __init__(
        self,
        cache_factory: Callable[[Sequence[int]], Any],
        single_factory: Callable[[], Any],
        merge_type: type,
        *,
        profile: StorageProfile,
        batch_size: int,
        max_size: int | None = None,
    ):
        self.cache_factory = cache_factory
        self.single_factory = single_factory
        self.merge_type = merge_type
        self.profile = profile
        self.max_size = max_size
        self.cache = cache_factory([0] * batch_size)
        self.snapshots: dict[str, Any] = {}
        self.extracted = None

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.EXTRACT: self._extract,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.EXTEND: self._extend,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(f"batch adapter does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        if self.extracted is not None:
            rows = [self.extracted]
        elif self.cache.empty():
            rows = [None] * self.cache.batch_size
        else:
            rows = self._extract_rows()
        keys = []
        values = []
        offsets = []
        positions = []
        key_shapes = []
        value_shapes = []
        key_dtype = None
        value_dtype = None
        pending_window = getattr(self.cache, "_lengths", None) is not None
        for row in rows:
            if row is None:
                keys.append(())
                values.append(())
                offsets.append(0)
                positions.append(())
                key_shapes.append(())
                value_shapes.append(())
                continue
            row_keys = getattr(row, "keys", None)
            row_values = getattr(row, "values", None)
            offset = int(row.offset)
            if row_keys is None or row_values is None:
                frozen_keys = ()
                frozen_values = ()
                key_shape = ()
                value_shape = ()
            else:
                if self.max_size is not None and not pending_window:
                    row_keys = row_keys[..., -self.max_size :, :]
                    row_values = row_values[..., -self.max_size :, :]
                import mlx.core as mx

                mx.eval(row_keys, row_values)
                frozen_keys = _freeze(row_keys[0].tolist())
                frozen_values = _freeze(row_values[0].tolist())
                key_shape = tuple(row_keys.shape[1:])
                value_shape = tuple(row_values.shape[1:])
                key_dtype = _dtype_name(row_keys.dtype)
                value_dtype = _dtype_name(row_values.dtype)
            length = _time_size(frozen_keys)
            keys.append(frozen_keys)
            values.append(frozen_values)
            offsets.append(offset)
            positions.append(tuple(range(offset - length, offset)))
            key_shapes.append(key_shape)
            value_shapes.append(value_shape)
        return CacheObservation(
            logical_keys=tuple(keys),
            logical_values=tuple(values),
            visible_positions=tuple(positions),
            offset=tuple(offsets),
            size=tuple(_time_size(row) for row in keys),
            shape=(tuple(key_shapes), tuple(value_shapes)),
            dtype=(key_dtype, value_dtype),
            batch_size=len(rows),
            metadata={
                "profile": self.profile.value,
                "max_size": self.max_size,
            },
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        self.extracted = None
        key_dtype = _mlx_dtype(
            payload.get("key_dtype", payload.get("dtype", "float32"))
        )
        value_dtype = _mlx_dtype(
            payload.get("value_dtype", payload.get("dtype", "float32"))
        )
        fetched = self.cache.update_and_fetch(
            mx.array(payload.get("keys"), dtype=key_dtype),
            mx.array(payload.get("values"), dtype=value_dtype),
        )
        mx.eval(*fetched)

    def _trim(self, payload: Mapping[str, Any]) -> None:
        self.extracted = None
        self.cache.trim(_non_negative_integer(payload.get("count"), "count"))

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.extracted = None
        self.cache = self.cache_factory([0] * self.cache.batch_size)

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshots[_snapshot_name(payload)] = _detach(
            self.cache.prefix_cache_snapshot()
        )

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self.extracted = None
        self.cache = self.cache_factory([0] * self.cache.batch_size)
        self.cache.prefix_cache_restore(_detach(self.snapshots[name]))

    def _extract(self, payload: Mapping[str, Any]) -> None:
        self.extracted = self.cache.extract(
            _normalized_index(payload.get("index"), self.cache.batch_size)
        )

    def _merge(self, payload: Mapping[str, Any]) -> None:
        self.extracted = None
        self.cache = self.merge_type.merge(
            [self._single_cache(row) for row in _rows(payload)]
        )

    def _extend(self, payload: Mapping[str, Any]) -> None:
        self.extracted = None
        rows = _rows(payload)
        other = self.merge_type.merge([self._single_cache(row) for row in rows])
        self.cache.extend(other)

    def _filter(self, payload: Mapping[str, Any]) -> None:
        self.extracted = None
        self.cache.filter(payload.get("indices"))

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        self.extracted = None
        self.cache.prepare(
            left_padding=payload.get("left_padding"),
            lengths=payload.get("lengths"),
            right_padding=payload.get("right_padding"),
        )

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        self.extracted = None
        self.cache.finalize()

    def _extract_rows(self) -> list[Any]:
        return [self.cache.extract(index) for index in range(self.cache.batch_size)]

    def _single_cache(self, row: Mapping[str, Any]) -> Any:
        import mlx.core as mx

        cache = self.single_factory()
        key_dtype = _mlx_dtype(row.get("key_dtype", row.get("dtype", "float32")))
        value_dtype = _mlx_dtype(row.get("value_dtype", row.get("dtype", "float32")))
        fetched = cache.update_and_fetch(
            mx.array([row["keys"]], dtype=key_dtype),
            mx.array([row["values"]], dtype=value_dtype),
        )
        mx.eval(*fetched)
        return cache


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("rows must be a non-empty sequence")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be an object")
        keys = _freeze(row.get("keys"))
        values = _freeze(row.get("values"))
        if len(keys) != 1 or len(values) != 1:
            raise ValueError("each merged row must have batch size one")
        normalized.append({**row, "keys": keys[0], "values": values[0]})
    return normalized


def _require_batch_values(values: Any, size: int, name: str) -> None:
    if not isinstance(values, Sequence) or len(values) != size:
        raise ValueError(f"{name} must match the batch size")


def _normalized_index(value: Any, size: int) -> int:
    index = _integer(value, "index")
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise ValueError("batch index is out of range")
    return index


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        return ()
    shape = []
    current = value
    while isinstance(current, tuple):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _row_layout(rows: Any) -> tuple[int, int] | None:
    shape = _shape(rows)
    if len(shape) != 4:
        return None
    return shape[1], shape[3]


def _row_shape(value: Any, layout: tuple[int, int] | None) -> tuple[int, ...]:
    shape = _shape(value)
    if len(shape) == 2 and shape[1] == 0 and layout is not None:
        return layout[0], 0, layout[1]
    return shape


def _time_size(value: Any) -> int:
    shape = _shape(value)
    return shape[1] if len(shape) >= 2 else 0


def _concatenate_time(left: Any, right: Any) -> Any:
    if not left:
        return right
    return tuple(
        tuple(left_head) + tuple(right_head)
        for left_head, right_head in zip(left, right)
    )


def _slice_time(value: Any, stop: int) -> Any:
    if stop == 0:
        return value
    return tuple(tuple(head[:stop]) for head in value)


def _slice_time_range(value: Any, start: int, stop: int) -> Any:
    return tuple(tuple(head[start:stop]) for head in value)


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
        return mx.array(value.tolist(), dtype=value.dtype)
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


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
