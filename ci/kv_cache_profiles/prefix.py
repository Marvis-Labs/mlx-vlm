from __future__ import annotations

from dataclasses import replace
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
from ci.kv_cache_profiles.common import cache_update

CAPABILITIES = frozenset(
    {
        CacheCapability.UPDATE,
        CacheCapability.TRIM,
        CacheCapability.RESET,
        CacheCapability.SNAPSHOT_RESTORE,
        CacheCapability.MERGE,
        CacheCapability.RESERVE,
    }
)


class PrefixOracle:
    """Independent logical prefix cache with capacity and read-only semantics."""

    capabilities = CAPABILITIES

    def __init__(self, kind: str, max_size: int, step: int):
        self.kind = kind
        self.initial_max_size = max_size
        self.step = step
        self.dense = DenseKVOracle()
        self.max_size = max_size
        self.allocated_capacity = 0
        self.read_only = False
        self.last_fetch: tuple[Any, Any] = ((), ())
        self.merge_supported: bool | None = None
        self.snapshots: dict[str, tuple[int, int, bool]] = {}

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.RESERVE: self._reserve,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(f"prefix oracle does not support {operation.kind.value}")
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        observation = self.dense.observe()
        metadata = {
            "kind": self.kind,
            "capacity": self.allocated_capacity,
            "max_size": self.max_size,
            "step": self.step,
            "read_only": self.read_only,
            "last_fetch": self.last_fetch,
            "merge_supported": self.merge_supported,
        }
        return replace(observation, metadata=metadata)

    def _update(self, payload: Mapping[str, Any]) -> None:
        action = payload.get("action", "append")
        update = CacheOperation(
            CacheOperationKind.UPDATE,
            {key: value for key, value in payload.items() if key != "action"},
        )
        if action == "fork_read_only":
            if self.kind != "static":
                raise ValueError("only static prefix caches support read-only forks")
            self.read_only = True
        elif action != "append":
            raise ValueError(f"unsupported prefix update action: {action}")
        if self.read_only:
            current = self.dense.observe()
            keys = _freeze(payload.get("keys"))
            values = _freeze(payload.get("values"))
            self.last_fetch = (
                _append_time(current.logical_keys, keys),
                _append_time(current.logical_values, values),
            )
            return
        observation = self.dense.apply(update)
        needed = int(observation.offset)
        if self.kind == "static":
            if needed > self.allocated_capacity:
                self.allocated_capacity = self._static_capacity(needed)
                self.max_size = self.allocated_capacity
        elif needed > self.allocated_capacity:
            update_length = _time_length(payload.get("keys"))
            blocks = (self.step + update_length - 1) // self.step
            self.allocated_capacity = needed - update_length + blocks * self.step
            self.max_size = self.allocated_capacity
        self.last_fetch = (observation.logical_keys, observation.logical_values)

    def _trim(self, payload: Mapping[str, Any]) -> None:
        observation = self.dense.apply(CacheOperation(CacheOperationKind.TRIM, payload))
        self.last_fetch = (observation.logical_keys, observation.logical_values)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.dense = DenseKVOracle()
        self.max_size = self.initial_max_size
        self.allocated_capacity = 0
        self.read_only = False
        self.last_fetch = ((), ())
        self.merge_supported = None

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self.dense.apply(CacheOperation(CacheOperationKind.SNAPSHOT, {"name": name}))
        self.snapshots[name] = (
            self.max_size,
            self.allocated_capacity,
            self.read_only,
        )
        self.last_fetch = ((), ())

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self.dense.apply(CacheOperation(CacheOperationKind.RESTORE, {"name": name}))
        self.max_size, _, self.read_only = self.snapshots[name]
        self.allocated_capacity = int(self.dense.observe().offset)
        if self.kind == "kv":
            self.max_size = self.allocated_capacity
        self.last_fetch = ((), ())
        self.merge_supported = None

    def _merge(self, payload: Mapping[str, Any]) -> None:
        prefix_lens = payload.get("prefix_lens")
        if not isinstance(prefix_lens, list):
            raise ValueError("prefix_lens must be a list")
        self.merge_supported = False
        self.last_fetch = ((), ())

    def _reserve(self, payload: Mapping[str, Any]) -> None:
        minimum = _non_negative_integer(payload.get("minimum"), "minimum")
        if self.kind == "kv" and self.allocated_capacity:
            self.allocated_capacity = max(
                self.allocated_capacity,
                _round_up(max(int(self.dense.observe().offset), minimum), self.step),
            )
            self.max_size = self.allocated_capacity
        self.last_fetch = ((), ())

    def _static_capacity(self, needed: int) -> int:
        if needed <= self.max_size:
            return self.max_size
        return _round_up(needed, self.step)


class MLXPrefixCacheAdapter:
    """Normalize StaticPrefixKVCache and KVCache prefix protocol behavior."""

    capabilities = CAPABILITIES

    def __init__(
        self,
        kind: str,
        cache_factory: Callable[[], Any],
    ):
        self.kind = kind
        self.cache_factory = cache_factory
        self.cache = cache_factory()
        self.snapshots: dict[str, Any] = {}
        self.last_fetch: tuple[Any, Any] = ((), ())
        self.merge_supported: bool | None = None

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.TRIM: self._trim,
            CacheOperationKind.RESET: self._reset,
            CacheOperationKind.SNAPSHOT: self._snapshot,
            CacheOperationKind.RESTORE: self._restore,
            CacheOperationKind.MERGE: self._merge,
            CacheOperationKind.RESERVE: self._reserve,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"prefix cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        keys, values = self.cache.state
        offset = _offset(self.cache)
        if keys is None or values is None:
            logical_keys = logical_values = ()
            shape = (None, None)
            dtype = (None, None)
            batch_size = 0
            allocated_capacity = 0
        else:
            mx.eval(keys, values)
            logical_keys = _freeze(keys.tolist())
            logical_values = _freeze(values.tolist())
            shape = (tuple(keys.shape), tuple(values.shape))
            dtype = (_dtype_name(keys.dtype), _dtype_name(values.dtype))
            batch_size = int(keys.shape[0])
            decoder_keys = getattr(self.cache, "keys", None)
            allocated_capacity = (
                0 if decoder_keys is None else int(decoder_keys.shape[2])
            )
        if self.kind == "static":
            max_size = int(self.cache.max_size)
            step = int(self.cache.step)
            read_only = bool(self.cache.read_only)
        else:
            max_size = allocated_capacity
            step = int(self.cache.step)
            read_only = False
        return CacheObservation(
            logical_keys=logical_keys,
            logical_values=logical_values,
            visible_positions=tuple(range(offset)),
            offset=offset,
            size=offset,
            shape=shape,
            dtype=dtype,
            batch_size=batch_size,
            metadata={
                "kind": self.kind,
                "capacity": allocated_capacity,
                "max_size": max_size,
                "step": step,
                "read_only": read_only,
                "last_fetch": self.last_fetch,
                "merge_supported": self.merge_supported,
            },
            allocated_bytes=int(self.cache.nbytes),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        action = payload.get("action", "append")
        if action == "fork_read_only":
            if self.kind != "static":
                raise ValueError("only static prefix caches support read-only forks")
            from mlx_vlm.models.cache import StaticPrefixKVCache

            self.cache = StaticPrefixKVCache.from_prefix(self.cache)
        elif action != "append":
            raise ValueError(f"unsupported prefix update action: {action}")
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
        self.last_fetch = (_freeze(fetched[0].tolist()), _freeze(fetched[1].tolist()))

    def _trim(self, payload: Mapping[str, Any]) -> None:
        self.cache.trim(_non_negative_integer(payload.get("count"), "count"))
        self.last_fetch = _state_values(self.cache)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.cache = self.cache_factory()
        self.last_fetch = ((), ())
        self.merge_supported = None

    def _snapshot(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        self.snapshots[name] = _detach(self.cache.prefix_cache_snapshot())
        self.last_fetch = ((), ())

    def _restore(self, payload: Mapping[str, Any]) -> None:
        name = _snapshot_name(payload)
        if name not in self.snapshots:
            raise ValueError(f"unknown snapshot: {name}")
        self.cache = self.cache_factory()
        self.cache.prefix_cache_restore(self.snapshots[name])
        self.last_fetch = ((), ())
        self.merge_supported = None

    def _merge(self, payload: Mapping[str, Any]) -> None:
        result = self.cache.prefix_cache_merge([self.cache], payload.get("prefix_lens"))
        self.merge_supported = result is not None
        self.last_fetch = ((), ())

    def _reserve(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        targets = self.cache.prefix_cache_reserve(
            _non_negative_integer(payload.get("minimum"), "minimum")
        )
        arrays = _arrays(targets)
        if arrays:
            mx.eval(*arrays)
        self.last_fetch = ((), ())


def prefix_contract_cases() -> tuple[CacheContractCase, ...]:
    from mlx_vlm.models.cache import KVCache, StaticPrefixKVCache

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
            name="StaticPrefixKVCache",
            profile=StorageProfile.PREFIX,
            subject_factory=lambda: MLXPrefixCacheAdapter(
                "static", lambda: StaticPrefixKVCache(max_size=8, step=4)
            ),
            oracle_factory=lambda: PrefixOracle("static", max_size=8, step=4),
            capabilities=CAPABILITIES,
            characteristics=characteristics,
            sequences=_static_sequences(),
        ),
        CacheContractCase(
            name="KVCachePrefixProtocol",
            profile=StorageProfile.PREFIX,
            subject_factory=lambda: MLXPrefixCacheAdapter("kv", KVCache),
            oracle_factory=lambda: PrefixOracle("kv", max_size=0, step=256),
            capabilities=CAPABILITIES,
            characteristics=characteristics,
            sequences=_kv_sequences(),
        ),
    )


def _static_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "fixed-capacity-overflow",
            (cache_update(0, 3), cache_update(3, 6)),
        ),
        OperationSequence(
            "trim-and-resume",
            (
                cache_update(0, 6),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                cache_update(4, 2),
            ),
        ),
        OperationSequence(
            "mutable-snapshot-continuation",
            (
                cache_update(0, 5, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prefix"}),
                cache_update(5, 2, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prefix"}),
                cache_update(5, 1, dtype="float16"),
            ),
        ),
        OperationSequence(
            "read-only-prefix",
            (
                cache_update(0, 3),
                _fork_update(20, 2),
                cache_update(30, 1),
            ),
        ),
        OperationSequence(
            "read-only-snapshot-restore",
            (
                cache_update(0, 3),
                _fork_update(20, 1),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "shared"}),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "shared"}),
                cache_update(30, 2),
            ),
        ),
        OperationSequence(
            "reserve-and-merge-refusal",
            (
                cache_update(0, 3),
                CacheOperation(CacheOperationKind.RESERVE, {"minimum": 20}),
                CacheOperation(CacheOperationKind.MERGE, {"prefix_lens": [2, 3]}),
            ),
        ),
    )


def _kv_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "empty-snapshot-restore",
            (
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "empty"}),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "empty"}),
                cache_update(0, 2),
            ),
        ),
        OperationSequence(
            "restore-reserve-continuation",
            (
                cache_update(0, 5, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "prompt"}),
                cache_update(5, 3, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "prompt"}),
                CacheOperation(CacheOperationKind.RESERVE, {"minimum": 600}),
                cache_update(5, 2, dtype="float16"),
            ),
        ),
        OperationSequence(
            "trim-snapshot-resume",
            (
                cache_update(0, 6),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "trimmed"}),
                cache_update(4, 2),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "trimmed"}),
                cache_update(4, 1),
            ),
        ),
        OperationSequence(
            "merge-refusal",
            (
                cache_update(0, 3),
                CacheOperation(CacheOperationKind.MERGE, {"prefix_lens": [2, 3]}),
            ),
        ),
    )


def _fork_update(start: int, count: int) -> CacheOperation:
    operation = cache_update(start, count)
    return CacheOperation(
        CacheOperationKind.UPDATE,
        {**operation.payload, "action": "fork_read_only"},
    )


def _state_values(cache: Any) -> tuple[Any, Any]:
    import mlx.core as mx

    keys, values = cache.state
    if keys is None or values is None:
        return (), ()
    mx.eval(keys, values)
    return _freeze(keys.tolist()), _freeze(values.tolist())


def _append_time(current: Any, update: Any) -> Any:
    if not current:
        return update
    return tuple(
        tuple(left_head + right_head for left_head, right_head in zip(left, right))
        for left, right in zip(current, update)
    )


def _time_length(value: Any) -> int:
    frozen = _freeze(value)
    return len(frozen[0][0])


def _round_up(value: int, step: int) -> int:
    return (value + step - 1) // step * step


def _offset(cache: Any) -> int:
    return int(cache.offset)


def _arrays(value: Any) -> list[Any]:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        arrays: list[Any] = []
        for item in value:
            arrays.extend(_arrays(item))
        return arrays
    return []


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
