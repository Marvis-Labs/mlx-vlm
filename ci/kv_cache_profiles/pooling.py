from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

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
        CacheCapability.RESET,
        CacheCapability.FILTER,
        CacheCapability.BATCH_LIFECYCLE,
    }
)


class PoolingOracle:
    """Pure-Python oracle for pooled tokens and incomplete-window buffering."""

    capabilities = SINGLE_CAPABILITIES

    def __init__(self, ratio: int):
        self.ratio = ratio
        self.buffer_kv: list[list[list[float]]] | None = None
        self.buffer_gate: list[list[list[float]]] | None = None
        self.pooled: list[list[list[float]]] | None = None
        self.dtype: str | None = None
        self.last_emitted = ((), (), 0)
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
            raise ValueError(f"pooling oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        remainder = 0 if self.buffer_kv is None else len(self.buffer_kv[0])
        pool_length = 0 if self.pooled is None else len(self.pooled[0])
        buffers = _freeze(self.buffer_kv) if remainder else ()
        gates = _freeze(self.buffer_gate) if remainder else ()
        pooled = _freeze(self.pooled or [])
        batch_size = _batch_size(
            self.buffer_kv if remainder else None,
            self.pooled,
        )
        return CacheObservation(
            logical_keys=(buffers, pooled),
            logical_values=gates,
            offset=pool_length,
            size=remainder,
            shape={
                "buffer": remainder,
                "pooled": pool_length,
            },
            dtype=self.dtype if remainder or pool_length else None,
            batch_size=batch_size,
            metadata={
                "ratio": self.ratio,
                "emitted": self.last_emitted,
                "empty": pool_length == 0 and remainder == 0,
            },
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        action = payload.get("action")
        if action == "accumulate":
            self._accumulate(payload)
        elif action == "pool":
            values = _nested_lists(payload.get("values"), "values")
            self.dtype = _dtype(payload)
            if not values or not values[0]:
                self.last_emitted = ((), (), 0)
                return
            if self.pooled is None:
                self.pooled = values
            else:
                self.pooled = [left + right for left, right in zip(self.pooled, values)]
            self.last_emitted = ((), (), 0)
        else:
            raise ValueError("pooling update action must be accumulate or pool")

    def _accumulate(self, payload: Mapping[str, Any]) -> None:
        kv = _nested_lists(payload.get("kv"), "kv")
        gate = _nested_lists(payload.get("gate"), "gate")
        if len(kv) != len(gate):
            raise ValueError("kv and gate batch sizes differ")
        self.dtype = _dtype(payload)
        if self.buffer_kv is None:
            self.buffer_kv = [[] for _ in kv]
            self.buffer_gate = [[] for _ in gate]
        if len(self.buffer_kv) != len(kv):
            raise ValueError("pooling batch size changed")
        old_remainder = len(self.buffer_kv[0])
        combined_kv = [left + right for left, right in zip(self.buffer_kv, kv)]
        combined_gate = [left + right for left, right in zip(self.buffer_gate, gate)]
        usable = len(combined_kv[0]) // self.ratio * self.ratio
        emitted_kv = [row[:usable] for row in combined_kv]
        emitted_gate = [row[:usable] for row in combined_gate]
        self.buffer_kv = [row[usable:] for row in combined_kv]
        self.buffer_gate = [row[usable:] for row in combined_gate]
        base = (
            _integer(payload.get("offset"), "offset") - old_remainder if usable else 0
        )
        self.last_emitted = (_freeze(emitted_kv), _freeze(emitted_gate), base)

    def _trim(self, payload: Mapping[str, Any]) -> None:
        count = min(
            int(self.observe().size),
            _non_negative_integer(payload.get("count"), "count"),
        )
        if count and self.buffer_kv is not None and self.buffer_gate is not None:
            self.buffer_kv = [row[:-count] for row in self.buffer_kv]
            self.buffer_gate = [row[:-count] for row in self.buffer_gate]
        self.last_emitted = ((), (), 0)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.buffer_kv = None
        self.buffer_gate = None
        self.pooled = None
        self.dtype = None
        self.last_emitted = ((), (), 0)

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshots[_snapshot_name(payload)] = deepcopy(
            (self.buffer_kv, self.buffer_gate, self.pooled, self.dtype)
        )
        self.last_emitted = ((), (), 0)

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self.buffer_kv, self.buffer_gate, self.pooled, self.dtype = deepcopy(
            self.snapshots[name]
        )
        self.last_emitted = ((), (), 0)


class BatchPoolingOracle:
    """Row-wise oracle for padded and variable-length pooling batches."""

    capabilities = BATCH_CAPABILITIES

    def __init__(self, ratio: int, left_padding: Sequence[int]):
        self.ratio = ratio
        self.left_padding = [int(value) for value in left_padding]
        size = len(self.left_padding)
        self.buffers_kv: list[list[list[float]]] = [[] for _ in range(size)]
        self.buffers_gate: list[list[list[float]]] = [[] for _ in range(size)]
        self.pooled: list[list[list[float]]] = [[] for _ in range(size)]
        self.lengths = [2**31] * size
        self.processed = [0] * size
        self.dtype: str | None = None
        self.last_emitted = ((), (), ())

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"batch pooling oracle does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        remainders = tuple(len(row) for row in self.buffers_kv)
        pool_lengths = tuple(len(row) for row in self.pooled)
        return CacheObservation(
            logical_keys=(_freeze(self.buffers_kv), _freeze(self.pooled)),
            logical_values=_freeze(self.buffers_gate),
            offset=pool_lengths,
            size=remainders,
            shape={"buffer": remainders, "pooled": pool_lengths},
            dtype=self.dtype,
            batch_size=len(self.left_padding),
            metadata={
                "ratio": self.ratio,
                "left_padding": tuple(self.left_padding),
                "processed": tuple(self.processed),
                "emitted": self.last_emitted,
                "empty": not any(remainders) and not any(pool_lengths),
            },
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        action = payload.get("action")
        if action == "accumulate":
            self._accumulate(payload)
        elif action == "pool":
            values = _nested_lists(payload.get("values"), "values")
            new_counts = [
                (processed - len(buffer)) // self.ratio - len(pooled)
                for processed, buffer, pooled in zip(
                    self.processed, self.buffers_kv, self.pooled
                )
            ]
            for row, count in enumerate(new_counts):
                self.pooled[row].extend(values[row][:count])
            self.dtype = _dtype(payload)
            self.last_emitted = ((), (), ())
        else:
            raise ValueError("pooling update action must be accumulate or pool")

    def _accumulate(self, payload: Mapping[str, Any]) -> None:
        kv = _nested_lists(payload.get("kv"), "kv")
        gate = _nested_lists(payload.get("gate"), "gate")
        offsets = payload.get("offset")
        if isinstance(offsets, int):
            offsets = [offsets] * len(kv)
        if not isinstance(offsets, Sequence) or len(offsets) != len(kv):
            raise ValueError("offset must match pooling batch size")
        emitted_kv: list[list[list[float]]] = []
        emitted_gate: list[list[list[float]]] = []
        bases: list[int] = []
        for row, (kv_row, gate_row) in enumerate(zip(kv, gate)):
            length = len(kv_row)
            start = min(self.left_padding[row], length)
            self.left_padding[row] -= start
            valid = max(
                0,
                min(self.lengths[row] - self.processed[row], length - start),
            )
            incoming_kv = kv_row[start : start + valid]
            incoming_gate = gate_row[start : start + valid]
            old_remainder = len(self.buffers_kv[row])
            combined_kv = self.buffers_kv[row] + incoming_kv
            combined_gate = self.buffers_gate[row] + incoming_gate
            usable = len(combined_kv) // self.ratio * self.ratio
            emitted_kv.append(combined_kv[:usable])
            emitted_gate.append(combined_gate[:usable])
            self.buffers_kv[row] = combined_kv[usable:]
            self.buffers_gate[row] = combined_gate[usable:]
            self.processed[row] += valid
            bases.append(int(offsets[row]) + start - old_remainder if usable else 0)
        width = max((len(row) for row in emitted_kv), default=0)
        emitted_kv = [_pad_rows(row, width, kv[0][0]) for row in emitted_kv]
        emitted_gate = [_pad_rows(row, width, gate[0][0]) for row in emitted_gate]
        self.dtype = _dtype(payload)
        self.last_emitted = (
            _freeze(emitted_kv),
            _freeze(emitted_gate),
            tuple(bases) if width else 0,
        )

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        size = len(self.left_padding)
        self.left_padding = [0] * size
        self.buffers_kv = [[] for _ in range(size)]
        self.buffers_gate = [[] for _ in range(size)]
        self.pooled = [[] for _ in range(size)]
        self.lengths = [2**31] * size
        self.processed = [0] * size
        self.dtype = None
        self.last_emitted = ((), (), ())

    def _filter(self, payload: Mapping[str, Any]) -> None:
        indices = [_integer(value, "index") for value in payload.get("indices", [])]
        for name in (
            "left_padding",
            "buffers_kv",
            "buffers_gate",
            "pooled",
            "lengths",
            "processed",
        ):
            values = getattr(self, name)
            setattr(self, name, [values[index] for index in indices])
        self.last_emitted = ((), (), ())

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        left_padding = payload.get("left_padding")
        if left_padding is not None:
            self.left_padding = [
                current + int(value)
                for current, value in zip(self.left_padding, left_padding)
            ]
        lengths = payload.get("lengths")
        if lengths is not None:
            self.lengths = [
                processed + int(length)
                for processed, length in zip(self.processed, lengths)
            ]
        self.last_emitted = ((), (), ())

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        self.lengths = [2**31] * len(self.left_padding)
        self.last_emitted = ((), (), ())


class MLXPoolingCacheAdapter:
    capabilities = SINGLE_CAPABILITIES

    def __init__(self, ratio: int):
        from mlx_vlm.models.cache import PoolingCache

        self.ratio = ratio
        self.cache = PoolingCache(ratio)
        self.last_emitted = ((), (), 0)
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
                f"pooling cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        state = self.cache.state
        arrays = [value for value in state if value is not None]
        if arrays:
            mx.eval(*arrays)
        buffer_kv, buffer_gate, pooled = (
            None if value is None else _nested_lists(value.tolist(), "state")
            for value in state
        )
        remainder = int(self.cache.remainder)
        pool_length = int(self.cache.size())
        return CacheObservation(
            logical_keys=(_freeze(buffer_kv or []), _freeze(pooled or [])),
            logical_values=_freeze(buffer_gate or []),
            offset=pool_length,
            size=remainder,
            shape={"buffer": remainder, "pooled": pool_length},
            dtype=_state_dtype(state),
            batch_size=_batch_size(buffer_kv, pooled),
            metadata={
                "ratio": int(self.cache.ratio),
                "emitted": self.last_emitted,
                "empty": self.cache.empty(),
            },
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        dtype = _mlx_dtype(_dtype(payload))
        action = payload.get("action")
        if action == "accumulate":
            result = self.cache.accumulate_windows(
                mx.array(payload.get("kv"), dtype=dtype),
                mx.array(payload.get("gate"), dtype=dtype),
                _integer(payload.get("offset"), "offset"),
            )
            mx.eval(result[0], result[1])
            self.last_emitted = (
                _freeze(result[0].tolist()),
                _freeze(result[1].tolist()),
                int(result[2]),
            )
        elif action == "pool":
            pooled = self.cache.update_and_fetch(
                mx.array(payload.get("values"), dtype=dtype)
            )
            mx.eval(pooled)
            self.last_emitted = ((), (), 0)
        else:
            raise ValueError("pooling update action must be accumulate or pool")

    def _trim(self, payload: Mapping[str, Any]) -> None:
        self.cache.trim(_non_negative_integer(payload.get("count"), "count"))
        self.last_emitted = ((), (), 0)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        from mlx_vlm.models.cache import PoolingCache

        if payload:
            raise ValueError("reset takes no payload")
        self.cache = PoolingCache(self.ratio)
        self.last_emitted = ((), (), 0)

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        self.snapshots[_snapshot_name(payload)] = _detach(
            self.cache.prefix_cache_snapshot()
        )
        self.last_emitted = ((), (), 0)

    def _restore(self, payload: Mapping[str, Any]) -> None:
        from mlx_vlm.models.cache import PoolingCache

        name = _snapshot_name(payload)
        self.cache = PoolingCache.from_state(
            self.snapshots[name]["state"], self.snapshots[name]["meta_state"]
        )
        self.last_emitted = ((), (), 0)


class MLXBatchPoolingCacheAdapter:
    capabilities = BATCH_CAPABILITIES

    def __init__(self, ratio: int, left_padding: Sequence[int]):
        from mlx_vlm.models.cache import BatchPoolingCache

        self.ratio = ratio
        self.initial_left_padding = list(left_padding)
        self.cache = BatchPoolingCache(ratio, list(left_padding))
        self.last_emitted = ((), (), ())

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"batch pooling adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        arrays = [
            value
            for value in (self.cache.buf_kv, self.cache.buf_gate, self.cache.pooled)
            if value is not None
        ]
        if arrays:
            mx.eval(*arrays)
        buffers_kv = _active_rows(self.cache.buf_kv, self.cache.remainder)
        buffers_gate = _active_rows(self.cache.buf_gate, self.cache.remainder)
        pooled = _active_rows(self.cache.pooled, self.cache._pool_lengths)
        remainders = tuple(int(value) for value in self.cache.remainder)
        pool_lengths = tuple(int(value) for value in self.cache._pool_lengths)
        return CacheObservation(
            logical_keys=(_freeze(buffers_kv), _freeze(pooled)),
            logical_values=_freeze(buffers_gate),
            offset=pool_lengths,
            size=remainders,
            shape={"buffer": remainders, "pooled": pool_lengths},
            dtype=_state_dtype(
                (self.cache.buf_kv, self.cache.buf_gate, self.cache.pooled)
            ),
            batch_size=len(self.cache.remainder),
            metadata={
                "ratio": int(self.cache.ratio),
                "left_padding": tuple(self.cache.left_padding),
                "processed": tuple(self.cache._processed),
                "emitted": self.last_emitted,
                "empty": self.cache.empty(),
            },
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        dtype = _mlx_dtype(_dtype(payload))
        action = payload.get("action")
        if action == "accumulate":
            offset = payload.get("offset")
            if isinstance(offset, Sequence):
                offset = mx.array(offset)
            result = self.cache.accumulate_windows(
                mx.array(payload.get("kv"), dtype=dtype),
                mx.array(payload.get("gate"), dtype=dtype),
                offset,
            )
            mx.eval(result[0], result[1])
            base = result[2]
            if isinstance(base, mx.array):
                mx.eval(base)
                base = tuple(int(value) for value in base.tolist())
            self.last_emitted = (
                _freeze(result[0].tolist()),
                _freeze(result[1].tolist()),
                base,
            )
        elif action == "pool":
            pooled = self.cache.update_and_fetch(
                mx.array(payload.get("values"), dtype=dtype)
            )
            mx.eval(pooled)
            self.last_emitted = ((), (), ())
        else:
            raise ValueError("pooling update action must be accumulate or pool")

    def _reset(self, payload: Mapping[str, Any]) -> None:
        from mlx_vlm.models.cache import BatchPoolingCache

        if payload:
            raise ValueError("reset takes no payload")
        self.cache = BatchPoolingCache(self.ratio, [0] * len(self.cache.remainder))
        self.last_emitted = ((), (), ())

    def _filter(self, payload: Mapping[str, Any]) -> None:
        self.cache.filter(payload.get("indices"))
        self.last_emitted = ((), (), ())

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        self.cache.prepare(
            lengths=payload.get("lengths"), left_padding=payload.get("left_padding")
        )
        self.last_emitted = ((), (), ())

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        self.cache.finalize()
        self.last_emitted = ((), (), ())


def pooling_contract_cases() -> tuple[CacheContractCase, ...]:
    characteristics = frozenset(
        {
            CacheCharacteristic.CONTENT,
            CacheCharacteristic.POSITION,
            CacheCharacteristic.SHAPE,
            CacheCharacteristic.DTYPE,
            CacheCharacteristic.BATCH_LAYOUT,
            CacheCharacteristic.METADATA,
        }
    )
    return (
        CacheContractCase(
            name="PoolingCache",
            profile=StorageProfile.POOLING,
            subject_factory=lambda: MLXPoolingCacheAdapter(4),
            oracle_factory=lambda: PoolingOracle(4),
            capabilities=SINGLE_CAPABILITIES,
            characteristics=characteristics,
            sequences=_single_sequences(),
        ),
        CacheContractCase(
            name="BatchPoolingCache",
            profile=StorageProfile.POOLING,
            subject_factory=lambda: MLXBatchPoolingCacheAdapter(4, [5, 0]),
            oracle_factory=lambda: BatchPoolingOracle(4, [5, 0]),
            capabilities=BATCH_CAPABILITIES,
            characteristics=characteristics,
            sequences=_batch_sequences(),
        ),
    )


def _single_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "prompt-decode-pool",
            (
                _accumulate(0, 3),
                _accumulate(3, 2),
                _pool(100, 1),
                _accumulate(5, 1),
                _accumulate(6, 2),
                _pool(101, 1),
            ),
        ),
        OperationSequence(
            "trim-remainder",
            (
                _accumulate(0, 3),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                _accumulate(1, 3),
                _pool(200, 1),
            ),
        ),
        OperationSequence(
            "snapshot-restore",
            (
                _accumulate(0, 3),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "partial"}),
                _accumulate(3, 2),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "partial"}),
                _accumulate(3, 1),
                _pool(300, 1),
            ),
        ),
    )


def _batch_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "left-padding-and-lengths",
            (
                CacheOperation(CacheOperationKind.PREPARE_BATCH, {"lengths": [4, 9]}),
                _accumulate(0, 3, batch_size=2, offsets=[-5, 0]),
                _accumulate(3, 3, batch_size=2, offsets=[-2, 3]),
                _accumulate(6, 3, batch_size=2, offsets=[1, 6]),
                _pool(500, 2, batch_size=2),
                CacheOperation(CacheOperationKind.FINALIZE_BATCH),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [1]}),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                _accumulate(0, 3, batch_size=2, offsets=[-5, 0]),
                CacheOperation(CacheOperationKind.RESET),
                _accumulate(20, 2, batch_size=2, offsets=[0, 0]),
            ),
        ),
    )


def _accumulate(
    start: int,
    count: int,
    batch_size: int = 1,
    offsets: Sequence[int] | None = None,
) -> CacheOperation:
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {
            "action": "accumulate",
            "kv": _values(start, count, batch_size, 2, 0),
            "gate": _values(start, count, batch_size, 1, 1000),
            "offset": start if offsets is None else list(offsets),
            "dtype": "float32",
        },
    )


def _pool(start: int, count: int, batch_size: int = 1) -> CacheOperation:
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {
            "action": "pool",
            "values": _values(start, count, batch_size, 2, 2000),
            "dtype": "float32",
        },
    )


def _values(
    start: int, count: int, batch_size: int, channels: int, family: int
) -> list[list[list[float]]]:
    return [
        [
            [
                float(family + batch * 100 + (start + position) * 2 + channel)
                for channel in range(channels)
            ]
            for position in range(count)
        ]
        for batch in range(batch_size)
    ]


def _active_rows(value: Any, lengths: Sequence[int]) -> list[Any]:
    if value is None:
        return [[] for _ in lengths]
    return [value[index, :length].tolist() for index, length in enumerate(lengths)]


def _pad_rows(
    row: list[list[float]], width: int, sample: list[float]
) -> list[list[float]]:
    return row + [[0.0] * len(sample) for _ in range(width - len(row))]


def _nested_lists(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a nested sequence")
    return deepcopy(list(value))


def _batch_size(*values: Any) -> int:
    for value in values:
        if value:
            return len(value)
    return 0


def _state_dtype(values: Sequence[Any]) -> str | None:
    for value in values:
        if value is not None:
            return str(value.dtype).rsplit(".", 1)[-1]
    return None


def _dtype(payload: Mapping[str, Any]) -> str:
    value = payload.get("dtype", "float32")
    if not isinstance(value, str):
        raise ValueError("dtype must be a string")
    return value


def _mlx_dtype(value: str) -> Any:
    import mlx.core as mx

    if not hasattr(mx, value):
        raise ValueError(f"unsupported MLX dtype: {value}")
    return getattr(mx, value)


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
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
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
