import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache


def _arrays(batch_size, tokens):
    keys = mx.arange(batch_size * 2 * tokens * 2).reshape(batch_size, 2, tokens, 2)
    values = mx.arange(batch_size * 2 * tokens * 3).reshape(batch_size, 2, tokens, 3)
    return keys, values


def test_batch_kv_mask_preserves_per_row_left_padding_and_causality():
    padding = [2, 0, 1]
    cache = BatchKVCache([0, 0, 0])
    cache.prepare(left_padding=padding)
    cache.update_and_fetch(*_arrays(3, 6))

    mask = cache.make_mask(2, return_array=True)
    mx.eval(mask)

    expected = [
        [[[key >= left and key <= 6 + query for key in range(8)] for query in range(2)]]
        for left in padding
    ]
    assert mask.tolist() == expected


def test_batch_rotating_mask_limits_every_row_to_the_window_after_crossing():
    cache = BatchRotatingKVCache(8, [0, 0, 0])
    cache.prepare(left_padding=[2, 0, 1])
    cache.update_and_fetch(*_arrays(3, 12))

    mask = cache.make_mask(2, return_array=True)
    mx.eval(mask)

    assert mask.shape == (3, 1, 2, 9)
    assert mx.sum(mask, axis=-1).tolist() == [[[8, 8]]] * 3
