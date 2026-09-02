import pytest

from ci.components.model_path import resource_requirements
from ci.runner_selection import (
    Device,
    RunnerSelectionError,
    ordered_devices,
    required_disk_gib,
    required_memory_gib,
    select_device,
)


def checkpoint_job(weight_gib=1):
    checkpoint = {"weight": {"bytes": int(weight_gib * 2**30)}}
    required_memory, required_disk = resource_requirements(checkpoint)
    return {
        "required_memory_gib": required_memory,
        "required_disk_gib": required_disk,
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


def test_candidates_are_ordered_by_memory_before_configured_priority():
    devices = [
        Device("m5", "device-m5", 128),
        Device("mini-1", "device-mini-1", 16),
        Device("mid", "device-mid", 64),
        Device("mini-2", "device-mini-2", 16),
    ]

    assert [device.name for device in ordered_devices(checkpoint_job(), devices)] == [
        "mini-1",
        "mini-2",
        "mid",
        "m5",
    ]


def test_unavailable_small_devices_are_skipped():
    devices = [
        Device("busy-mini", "device-busy", 16, busy=True),
        Device("unhealthy-mini", "device-unhealthy", 16, healthy=False),
        Device("m5", "device-m5", 128),
    ]

    assert select_device(checkpoint_job(), devices).name == "m5"


def test_model_path_plugin_calculates_runtime_and_download_headroom():
    assert resource_requirements({"weight": {"bytes": 1 * 2**30}}) == (8, 4)
    assert resource_requirements({"weight": {"bytes": 40 * 2**30}}) == (64, 52)


def test_scheduler_reads_explicit_memory_requirement():
    assert required_memory_gib({"required_memory_gib": 32}) == 32


def test_scheduler_reads_explicit_disk_requirement():
    assert required_disk_gib({"required_disk_gib": 32}) == 32


def test_scheduler_rejects_missing_resource_requirements():
    with pytest.raises(RunnerSelectionError, match="required_memory_gib"):
        required_memory_gib({})
    with pytest.raises(RunnerSelectionError, match="required_disk_gib"):
        required_disk_gib({})


def test_selection_fails_when_no_device_can_fit():
    with pytest.raises(RunnerSelectionError, match="required 64 GiB"):
        select_device(checkpoint_job(40), [Device("mini", "device-mini", 16)])
