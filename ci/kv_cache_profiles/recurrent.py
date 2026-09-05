from __future__ import annotations

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

CAPABILITIES = frozenset(
    {
        CacheCapability.UPDATE,
        CacheCapability.RESET,
        CacheCapability.EXTRACT,
        CacheCapability.MERGE,
        CacheCapability.FILTER,
        CacheCapability.BATCH_LIFECYCLE,
        CacheCapability.ADVANCE,
    }
)


class ArraysCacheOracle:
    """Pure-Python reference for recurrent array-state and batch metadata."""

    capabilities = CAPABILITIES

    def __init__(self, size: int, left_padding: Sequence[int] | None = None):
        self.size = size
        self.states: list[Any | None] = [None] * size
        self.dtypes: list[str | None] = [None] * size
        self.left_padding = tuple(left_padding) if left_padding is not None else None
        self.lengths: tuple[int, ...] | None = None

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.EXTRACT: self._extract,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
            CacheOperationKind.ADVANCE: self._advance,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(f"arrays oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        shapes = tuple(
            None if state is None else _shape(state) for state in self.states
        )
        return CacheObservation(
            logical_keys=tuple(self.states),
            logical_values=None,
            visible_positions=tuple(state is not None for state in self.states),
            offset=self.lengths,
            size=self.left_padding,
            shape=shapes,
            dtype=tuple(self.dtypes),
            batch_size=self._batch_size(),
            metadata={
                "slots": self.size,
                "empty": all(state is None for state in self.states),
            },
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        slot = _slot(payload.get("slot"), self.size)
        state = _freeze(payload.get("state"))
        shape = _shape(state)
        if not shape:
            raise ValueError("recurrent state must have a batch axis")
        expected_batch = self._known_batch_size()
        if expected_batch and shape[0] != expected_batch:
            raise ValueError("recurrent state batch size is incompatible")
        self.states[slot] = state
        self.dtypes[slot] = _dtype(payload)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.states = [None] * self.size
        self.dtypes = [None] * self.size
        self.left_padding = None
        self.lengths = None

    def _extract(self, payload: Mapping[str, Any]) -> None:
        index = _normalized_index(
            _integer(payload.get("index"), "index"), self._batch_size()
        )
        self.states = [
            None if state is None else (state[index],) for state in self.states
        ]
        if self.left_padding is not None:
            self.left_padding = (self.left_padding[index],)
        if self.lengths is not None:
            self.lengths = (self.lengths[index],)

    def _merge(self, payload: Mapping[str, Any]) -> None:
        other = ArraysCacheOracle(self.size, payload.get("left_padding"))
        other_lengths = payload.get("lengths")
        if other_lengths is not None:
            other.lengths = tuple(int(value) for value in other_lengths)
        states = payload.get("states")
        if not isinstance(states, Sequence) or len(states) != self.size:
            raise ValueError("merge states must match recurrent slot count")
        dtypes = payload.get("dtypes", ["float32"] * self.size)
        for slot, state in enumerate(states):
            if state is not None:
                other._update({"slot": slot, "state": state, "dtype": dtypes[slot]})
        left_batch = self._batch_size()
        right_batch = other._batch_size()
        for slot, (left, right) in enumerate(zip(self.states, other.states)):
            self.states[slot] = _merge_state(left, right, left_batch, right_batch)
            if self.dtypes[slot] is None:
                self.dtypes[slot] = other.dtypes[slot]
        self.left_padding = _merge_vector(
            self.left_padding, other.left_padding, left_batch, right_batch
        )
        self.lengths = _merge_vector(
            self.lengths, other.lengths, left_batch, right_batch
        )

    def _filter(self, payload: Mapping[str, Any]) -> None:
        indices = _indices(payload.get("indices"), self._batch_size())
        self.states = [
            None if state is None else tuple(state[index] for index in indices)
            for state in self.states
        ]
        if self.left_padding is not None:
            self.left_padding = tuple(self.left_padding[index] for index in indices)
        if self.lengths is not None:
            self.lengths = tuple(self.lengths[index] for index in indices)

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        lengths = payload.get("lengths")
        if not isinstance(lengths, Sequence):
            raise ValueError("lengths must be a sequence")
        values = tuple(_integer(value, "length") for value in lengths)
        if self._known_batch_size() not in {0, len(values)}:
            raise ValueError("lengths must match recurrent batch size")
        self.lengths = values

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        self.lengths = None
        self.left_padding = None

    def _advance(self, payload: Mapping[str, Any]) -> None:
        count = _integer(payload.get("count"), "count")
        if self.lengths is not None:
            self.lengths = tuple(value - count for value in self.lengths)
        if self.left_padding is not None:
            self.left_padding = tuple(value - count for value in self.left_padding)

    def _batch_size(self) -> int:
        return self._known_batch_size() or 1

    def _known_batch_size(self) -> int:
        for state in self.states:
            if state is not None:
                return _shape(state)[0]
        if self.left_padding is not None:
            return len(self.left_padding)
        if self.lengths is not None:
            return len(self.lengths)
        return 0


class MLXArraysCacheAdapter:
    """Normalize ArraysCache state and lifecycle metadata for contract checks."""

    capabilities = CAPABILITIES

    def __init__(self, size: int, left_padding: Sequence[int] | None = None):
        from mlx_vlm.models.cache import ArraysCache

        self.size = size
        self.initial_left_padding = left_padding
        self.cache = ArraysCache(size, left_padding=left_padding)

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.EXTRACT: self._extract,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.FILTER: self._filter,
            CacheOperationKind.PREPARE_BATCH: self._prepare,
            CacheOperationKind.FINALIZE_BATCH: self._finalize,
            CacheOperationKind.ADVANCE: self._advance,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"arrays cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        arrays = [state for state in self.cache.cache if state is not None]
        metadata_arrays = [
            value
            for value in (self.cache.left_padding, self.cache.lengths)
            if value is not None
        ]
        if arrays or metadata_arrays:
            mx.eval(*(arrays + metadata_arrays))
        states = tuple(
            None if state is None else _freeze(state.tolist())
            for state in self.cache.cache
        )
        return CacheObservation(
            logical_keys=states,
            logical_values=None,
            visible_positions=tuple(state is not None for state in states),
            offset=_array_tuple(self.cache.lengths),
            size=_array_tuple(self.cache.left_padding),
            shape=tuple(
                None if state is None else tuple(state.shape)
                for state in self.cache.cache
            ),
            dtype=tuple(
                None if state is None else _dtype_name(state.dtype)
                for state in self.cache.cache
            ),
            batch_size=int(self.cache.batch_size),
            metadata={"slots": self.size, "empty": self.cache.empty()},
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        slot = _slot(payload.get("slot"), self.size)
        self.cache[slot] = mx.array(
            payload.get("state"), dtype=_mlx_dtype(_dtype(payload))
        )

    def _reset(self, payload: Mapping[str, Any]) -> None:
        from mlx_vlm.models.cache import ArraysCache

        if payload:
            raise ValueError("reset takes no payload")
        self.cache = ArraysCache(self.size)

    def _extract(self, payload: Mapping[str, Any]) -> None:
        self.cache = self.cache.extract(_integer(payload.get("index"), "index"))

    def _merge(self, payload: Mapping[str, Any]) -> None:
        from mlx_vlm.models.cache import ArraysCache

        other = ArraysCache(self.size, left_padding=payload.get("left_padding"))
        lengths = payload.get("lengths")
        if lengths is not None:
            other.prepare(lengths=lengths)
        states = payload.get("states")
        dtypes = payload.get("dtypes", ["float32"] * self.size)
        for slot, state in enumerate(states):
            if state is not None:
                import mlx.core as mx

                other[slot] = mx.array(state, dtype=_mlx_dtype(dtypes[slot]))
        self.cache.extend(other)

    def _filter(self, payload: Mapping[str, Any]) -> None:
        self.cache.filter(payload.get("indices"))

    def _prepare(self, payload: Mapping[str, Any]) -> None:
        self.cache.prepare(lengths=payload.get("lengths"))

    def _finalize(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("finalize takes no payload")
        self.cache.finalize()

    def _advance(self, payload: Mapping[str, Any]) -> None:
        self.cache.advance(_integer(payload.get("count"), "count"))


def recurrent_contract_cases() -> tuple[CacheContractCase, ...]:
    characteristics = frozenset(
        {
            CacheCharacteristic.CONTENT,
            CacheCharacteristic.VISIBILITY,
            CacheCharacteristic.POSITION,
            CacheCharacteristic.SHAPE,
            CacheCharacteristic.DTYPE,
            CacheCharacteristic.BATCH_LAYOUT,
            CacheCharacteristic.METADATA,
        }
    )
    return (
        CacheContractCase(
            name="ArraysCache",
            profile=StorageProfile.RECURRENT,
            subject_factory=lambda: MLXArraysCacheAdapter(3),
            oracle_factory=lambda: ArraysCacheOracle(3),
            capabilities=CAPABILITIES,
            characteristics=characteristics,
            sequences=_sequences(),
        ),
    )


def _sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "partial-slots-filter-extract",
            (
                _state_update(1, 0, 3),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [2, 0]}),
                CacheOperation(CacheOperationKind.EXTRACT, {"index": 1}),
            ),
        ),
        OperationSequence(
            "batch-lifecycle",
            (
                _state_update(0, 0, 3),
                CacheOperation(
                    CacheOperationKind.PREPARE_BATCH, {"lengths": [10, 7, 4]}
                ),
                CacheOperation(CacheOperationKind.ADVANCE, {"count": 3}),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [1, 0]}),
                CacheOperation(CacheOperationKind.FINALIZE_BATCH),
            ),
        ),
        OperationSequence(
            "sparse-slot-merge",
            (
                _state_update(1, 0, 2),
                CacheOperation(
                    CacheOperationKind.MERGE,
                    {
                        "states": [_state_values(50, 1), None, None],
                        "dtypes": ["float32", "float32", "float32"],
                    },
                ),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                _state_update(2, 0, 2, dtype="float16"),
                CacheOperation(CacheOperationKind.RESET),
                _state_update(0, 20, 1),
            ),
        ),
    )


def _state_update(
    slot: int, start: int, batch_size: int, dtype: str = "float32"
) -> CacheOperation:
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {"slot": slot, "state": _state_values(start, batch_size), "dtype": dtype},
    )


def _state_values(start: int, batch_size: int) -> list[list[list[float]]]:
    return [
        [
            [float(start + batch * 10 + row * 2 + channel) for channel in range(2)]
            for row in range(3)
        ]
        for batch in range(batch_size)
    ]


def _merge_state(left: Any, right: Any, left_batch: int, right_batch: int) -> Any:
    if left is None and right is None:
        return None
    sample = left if left is not None else right
    sample_shape = _shape(sample)
    if left is None:
        left = _zeros((left_batch,) + sample_shape[1:])
    if right is None:
        right = _zeros((right_batch,) + sample_shape[1:])
    return tuple(left) + tuple(right)


def _merge_vector(
    left: tuple[int, ...] | None,
    right: tuple[int, ...] | None,
    left_batch: int,
    right_batch: int,
) -> tuple[int, ...] | None:
    if left is None and right is None:
        return None
    return (left or (0,) * left_batch) + (right or (0,) * right_batch)


def _zeros(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return tuple(0.0 for _ in range(shape[0]))
    return tuple(_zeros(shape[1:]) for _ in range(shape[0]))


def _array_tuple(value: Any) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(item) for item in value.tolist())


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        return ()
    if not value:
        return (0,)
    child_shapes = {_shape(item) for item in value}
    if len(child_shapes) != 1:
        raise ValueError("recurrent state must be rectangular")
    return (len(value),) + child_shapes.pop()


def _dtype(payload: Mapping[str, Any]) -> str:
    dtype = payload.get("dtype", "float32")
    if not isinstance(dtype, str):
        raise ValueError("dtype must be a string")
    return dtype


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _mlx_dtype(value: str) -> Any:
    import mlx.core as mx

    if not hasattr(mx, value):
        raise ValueError(f"unsupported MLX dtype: {value}")
    return getattr(mx, value)


def _slot(value: Any, size: int) -> int:
    slot = _integer(value, "slot")
    if slot < 0 or slot >= size:
        raise IndexError(f"slot {slot} is out of range")
    return slot


def _indices(value: Any, batch_size: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence):
        raise ValueError("indices must be a sequence")
    return tuple(
        _normalized_index(_integer(index, "index"), batch_size) for index in value
    )


def _normalized_index(index: int, size: int) -> int:
    normalized = index + size if index < 0 else index
    if normalized < 0 or normalized >= size:
        raise IndexError(f"batch index {index} is out of range for size {size}")
    return normalized


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
