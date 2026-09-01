from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class RunnerSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Device:
    name: str
    label: str
    memory_gib: int
    online: bool = True
    busy: bool = False
    healthy: bool = True


def required_memory_gib(job: Mapping[str, Any]) -> int:
    configured = job.get("required_memory_gib")
    if not isinstance(configured, int) or configured <= 0:
        raise RunnerSelectionError("required_memory_gib must be a positive integer")
    return configured


def required_disk_gib(job: Mapping[str, Any]) -> int:
    configured = job.get("required_disk_gib")
    if not isinstance(configured, int) or configured <= 0:
        raise RunnerSelectionError("required_disk_gib must be a positive integer")
    return configured


def select_device(job: Mapping[str, Any], devices: Sequence[Device]) -> Device:
    eligible = ordered_devices(job, devices)
    if not eligible:
        required = required_memory_gib(job)
        raise RunnerSelectionError(
            f"no healthy idle device has the required {required} GiB"
        )
    return eligible[0]


def ordered_devices(
    job: Mapping[str, Any], devices: Sequence[Device]
) -> tuple[Device, ...]:
    required = required_memory_gib(job)
    eligible = [
        (index, device)
        for index, device in enumerate(devices)
        if device.online
        and not device.busy
        and device.healthy
        and device.memory_gib >= required
    ]
    return tuple(
        device
        for _, device in sorted(
            eligible, key=lambda item: (item[1].memory_gib, item[0])
        )
    )
