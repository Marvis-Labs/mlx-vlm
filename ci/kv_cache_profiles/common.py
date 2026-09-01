from __future__ import annotations

from ci.kv_cache_contract import CacheOperation, CacheOperationKind


def cache_update(
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
