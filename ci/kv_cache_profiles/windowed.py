from __future__ import annotations

import random
from typing import Any, Callable, Mapping

from ci.kv_cache_batch import BATCH_CAPABILITIES, BatchKVOracle, MLXBatchKVAdapter
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
from ci.kv_cache_oracles import WindowedKVOracle
from ci.kv_cache_profiles.common import cache_update as dense_update


class MLXWindowedCacheAdapter:
    """Expose an MLX sliding-window cache through semantic observations."""

    capabilities = frozenset({CacheCapability.UPDATE, CacheCapability.RESET})

    def __init__(self, cache_factory: Callable[[], Any]):
        self.cache_factory = cache_factory
        self.cache = cache_factory()

    def apply(self, operation: CacheOperation) -> CacheObservation:
        handlers = {
            CacheOperationKind.UPDATE: self._update,
            CacheOperationKind.RESET: self._reset,
        }
        handler = handlers.get(operation.kind)
        if handler is None:
            raise ValueError(
                f"windowed cache adapter does not support {operation.kind.value}"
            )
        handler(operation.payload)
        return self.observe()

    def observe(self) -> CacheObservation:
        import mlx.core as mx

        offset = int(self.cache.offset)
        keys = getattr(self.cache, "keys", None)
        values = getattr(self.cache, "values", None)
        max_size = int(self.cache.max_size)
        keep = int(self.cache.keep)
        metadata = {"max_size": max_size, "keep": keep}
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
                metadata=metadata,
                allocated_bytes=0,
            )
        active_keys, active_values = self._active(keys, values)
        mx.eval(active_keys, active_values)
        positions = self._positions(offset, max_size, keep)
        return CacheObservation(
            logical_keys=_freeze(active_keys.tolist()),
            logical_values=_freeze(active_values.tolist()),
            visible_positions=positions,
            offset=offset,
            size=len(positions),
            shape=(tuple(active_keys.shape), tuple(active_values.shape)),
            dtype=(_dtype_name(active_keys.dtype), _dtype_name(active_values.dtype)),
            batch_size=int(active_keys.shape[0]),
            metadata=metadata,
            allocated_bytes=int(self.cache.nbytes),
        )

    def _active(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        if hasattr(self.cache, "start_position") and not self.cache.keep:
            ordered_keys = keys[..., : self.cache._idx, :]
            ordered_values = values[..., : self.cache._idx, :]
        else:
            ordered_keys = self.cache._temporal_order(keys)
            ordered_values = self.cache._temporal_order(values)
        if self.cache.offset <= self.cache.max_size:
            return (
                ordered_keys[..., : self.cache.offset, :],
                ordered_values[..., : self.cache.offset, :],
            )
        recent = self.cache.max_size - self.cache.keep
        if not self.cache.keep:
            return ordered_keys[..., -recent:, :], ordered_values[..., -recent:, :]
        return (
            mx.concatenate(
                [
                    ordered_keys[..., : self.cache.keep, :],
                    ordered_keys[..., -recent:, :],
                ],
                axis=2,
            ),
            mx.concatenate(
                [
                    ordered_values[..., : self.cache.keep, :],
                    ordered_values[..., -recent:, :],
                ],
                axis=2,
            ),
        )

    def _update(self, payload: Mapping[str, Any]) -> None:
        import mlx.core as mx

        dtype = _mlx_dtype(payload.get("dtype", "float32"))
        keys = mx.array(payload.get("keys"), dtype=dtype)
        values = mx.array(payload.get("values"), dtype=dtype)
        fetched = self.cache.update_and_fetch(keys, values)
        mx.eval(*fetched)

    def _reset(self, payload: Mapping[str, Any]) -> None:
        if payload:
            raise ValueError("reset takes no payload")
        self.cache = self.cache_factory()

    @staticmethod
    def _positions(offset: int, max_size: int, keep: int) -> tuple[int, ...]:
        if offset <= max_size:
            return tuple(range(offset))
        return tuple(range(keep)) + tuple(range(offset - (max_size - keep), offset))


def windowed_contract_cases() -> tuple[CacheContractCase, ...]:
    from mlx_vlm.models.cache import (
        BatchRotatingKVCache,
        BufferedRotatingKVCache,
        RotatingKVCache,
    )

    capabilities = frozenset({CacheCapability.UPDATE, CacheCapability.RESET})
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
            name="RotatingKVCache",
            profile=StorageProfile.WINDOWED,
            subject_factory=lambda: MLXWindowedCacheAdapter(
                lambda: RotatingKVCache(max_size=8, keep=2)
            ),
            oracle_factory=lambda: WindowedKVOracle(max_size=8, keep=2),
            capabilities=capabilities,
            characteristics=characteristics,
            sequences=_sequences("rotating"),
        ),
        CacheContractCase(
            name="BufferedRotatingKVCache",
            profile=StorageProfile.WINDOWED,
            subject_factory=lambda: MLXWindowedCacheAdapter(
                lambda: BufferedRotatingKVCache(max_size=8, keep=0, buffer_size=4)
            ),
            oracle_factory=lambda: WindowedKVOracle(max_size=8, keep=0),
            capabilities=capabilities,
            characteristics=characteristics,
            sequences=_sequences("buffered"),
        ),
        CacheContractCase(
            name="BatchRotatingKVCache",
            profile=StorageProfile.WINDOWED,
            subject_factory=lambda: MLXBatchKVAdapter(
                lambda left_padding: BatchRotatingKVCache(8, list(left_padding)),
                lambda: RotatingKVCache(8),
                BatchRotatingKVCache,
                profile=StorageProfile.WINDOWED,
                batch_size=3,
                max_size=8,
            ),
            oracle_factory=lambda: BatchKVOracle(
                3, profile=StorageProfile.WINDOWED, max_size=8
            ),
            capabilities=BATCH_CAPABILITIES,
            characteristics=characteristics,
            sequences=_batch_sequences() + _batch_random_sequences(),
        ),
    )


def _sequences(prefix: str) -> tuple[OperationSequence, ...]:
    deterministic = (
        OperationSequence(
            "token-wrap",
            tuple(dense_update(index, 1) for index in range(12)),
        ),
        OperationSequence(
            "block-window-crossing",
            (
                dense_update(0, 5),
                dense_update(5, 5),
                dense_update(10, 3),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                dense_update(0, 10),
                CacheOperation(CacheOperationKind.RESET),
                dense_update(20, 3),
            ),
        ),
    )
    return deterministic + _random_sequences(prefix)


def _random_sequences(prefix: str) -> tuple[OperationSequence, ...]:
    sequences: list[OperationSequence] = []
    for seed in range(5):
        generator = random.Random(seed)
        cursor = 1000 + seed * 1000
        operations: list[CacheOperation] = []
        for _ in range(25):
            if generator.random() < 0.15:
                operations.append(CacheOperation(CacheOperationKind.RESET))
                continue
            count = generator.randint(1, 4)
            operations.append(dense_update(cursor, count))
            cursor += count
        sequences.append(
            OperationSequence(f"{prefix}-state-machine-{seed}", tuple(operations))
        )
    return tuple(sequences)


def _batch_sequences() -> tuple[OperationSequence, ...]:
    return (
        OperationSequence(
            "left-padding-prefill-decode",
            (
                CacheOperation(
                    CacheOperationKind.PREPARE_BATCH,
                    {"left_padding": [2, 0, 1]},
                ),
                dense_update(0, 6, batch_size=3),
                dense_update(6, 1, batch_size=3),
                dense_update(7, 1, batch_size=3),
                dense_update(8, 1, batch_size=3),
            ),
        ),
        OperationSequence(
            "right-padding-finalize-decode",
            (
                CacheOperation(
                    CacheOperationKind.PREPARE_BATCH,
                    {
                        "right_padding": [2, 0, 3],
                        "lengths": [4, 6, 3],
                    },
                ),
                dense_update(0, 6, batch_size=3),
                CacheOperation(CacheOperationKind.FINALIZE_BATCH),
                dense_update(6, 1, batch_size=3),
                dense_update(7, 1, batch_size=3),
            ),
        ),
        OperationSequence(
            "block-window-crossing",
            (
                dense_update(0, 5, batch_size=3),
                dense_update(5, 5, batch_size=3),
                dense_update(10, 3, batch_size=3),
            ),
        ),
        OperationSequence(
            "filter-and-extract",
            (
                dense_update(0, 10, batch_size=3),
                CacheOperation(CacheOperationKind.FILTER, {"indices": [2, 0]}),
                CacheOperation(CacheOperationKind.EXTRACT, {"index": 1}),
            ),
        ),
        OperationSequence(
            "merge-and-extend",
            (
                CacheOperation(
                    CacheOperationKind.MERGE,
                    {"rows": [_row(100, 2), _row(200, 8), _row(300, 5)]},
                ),
                CacheOperation(
                    CacheOperationKind.EXTEND,
                    {"rows": [_row(400, 7), _row(500, 3)]},
                ),
                dense_update(600, 1, batch_size=5),
            ),
        ),
        OperationSequence(
            "snapshot-restore-resume",
            (
                dense_update(0, 7, batch_size=3, dtype="float16"),
                CacheOperation(CacheOperationKind.SNAPSHOT, {"name": "window"}),
                dense_update(7, 4, batch_size=3, dtype="float16"),
                CacheOperation(CacheOperationKind.RESTORE, {"name": "window"}),
                dense_update(7, 1, batch_size=3, dtype="float16"),
            ),
        ),
        OperationSequence(
            "trim-before-saturation",
            (
                dense_update(0, 5, batch_size=3),
                CacheOperation(CacheOperationKind.TRIM, {"count": 2}),
                dense_update(3, 2, batch_size=3),
            ),
        ),
        OperationSequence(
            "reset-and-reuse",
            (
                dense_update(0, 10, batch_size=3),
                CacheOperation(CacheOperationKind.RESET),
                dense_update(20, 3, batch_size=3),
            ),
        ),
    )


def _batch_random_sequences() -> tuple[OperationSequence, ...]:
    sequences = []
    for seed in range(5):
        generator = random.Random(100 + seed)
        cursor = 1000 + seed * 1000
        operations = []
        for _ in range(25):
            if generator.random() < 0.15:
                operations.append(CacheOperation(CacheOperationKind.RESET))
                continue
            count = generator.randint(1, 4)
            operations.append(dense_update(cursor, count, batch_size=3))
            cursor += count
        sequences.append(
            OperationSequence(f"batch-window-state-machine-{seed}", tuple(operations))
        )
    return tuple(sequences)


def _row(start: int, count: int) -> dict[str, Any]:
    return dict(dense_update(start, count).payload)


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
