import pytest

from ci.runner_selection import (
    Device,
    RunnerSelectionError,
    required_memory_gib,
    select_device,
)


def checkpoint_job(weight_gib=1):
    return {
        "mode": "hf_checkpoint",
        "hf_checkpoint": {"weight": {"bytes": int(weight_gib * 2**30)}},
    }


def test_smallest_device_wins_even_when_largest_is_listed_first():
    devices = [
        Device("m5", "device-m5", 128),
        Device("mini-1", "device-mini-1", 16),
        Device("mini-2", "device-mini-2", 16),
    ]

    assert select_device(checkpoint_job(), devices).name == "mini-1"


def test_list_order_breaks_ties_between_same_size_devices():
    devices = [
        Device("mini-2", "device-mini-2", 16),
        Device("mini-1", "device-mini-1", 16),
    ]

    assert select_device(checkpoint_job(), devices).name == "mini-2"


def test_unavailable_small_devices_are_skipped():
    devices = [
        Device("busy-mini", "device-busy", 16, busy=True),
        Device("unhealthy-mini", "device-unhealthy", 16, healthy=False),
        Device("m5", "device-m5", 128),
    ]

    assert select_device(checkpoint_job(), devices).name == "m5"


def test_memory_requirement_uses_checkpoint_weight_and_runtime_headroom():
    assert required_memory_gib(checkpoint_job(1)) == 8
    assert required_memory_gib(checkpoint_job(40)) == 64


def test_explicit_memory_requirement_takes_precedence():
    assert required_memory_gib({"minimum_memory_gib": 32}) == 32


def test_selection_fails_when_no_device_can_fit():
    with pytest.raises(RunnerSelectionError, match="required 64 GiB"):
        select_device(checkpoint_job(40), [Device("mini", "device-mini", 16)])
