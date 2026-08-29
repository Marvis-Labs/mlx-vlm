from __future__ import annotations

import math
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
    configured = job.get("minimum_memory_gib")
    if configured is not None:
        if not isinstance(configured, int) or configured <= 0:
            raise RunnerSelectionError("minimum_memory_gib must be a positive integer")
        return configured

    if job.get("mode") == "synthetic":
        return 8

    checkpoint = job.get("hf_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RunnerSelectionError("job has no memory requirement")
    weight = checkpoint.get("weight")
    if not isinstance(weight, Mapping):
        raise RunnerSelectionError("checkpoint has no weight metadata")
    weight_bytes = weight.get("bytes")
    if not isinstance(weight_bytes, int) or weight_bytes <= 0:
        raise RunnerSelectionError("checkpoint weight bytes must be positive")

    weights_gib = weight_bytes / 2**30
    return max(8, math.ceil(weights_gib * 1.5 + 4))


def select_device(job: Mapping[str, Any], devices: Sequence[Device]) -> Device:
    required = required_memory_gib(job)
    eligible = [
        (index, device)
        for index, device in enumerate(devices)
        if device.online
        and not device.busy
        and device.healthy
        and device.memory_gib >= required
    ]
    if not eligible:
        raise RunnerSelectionError(
            f"no healthy idle device has the required {required} GiB"
        )
    return min(eligible, key=lambda item: (item[1].memory_gib, item[0]))[1]
