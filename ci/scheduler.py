from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ci.runner_selection import (
    Device,
    ordered_devices,
    required_disk_gib,
    required_memory_gib,
)


class DeviceDecision(str, Enum):
    ACCEPTED = "accepted"
    DECLINED = "declined"


class DeclineReason(str, Enum):
    BUSY = "declined_busy"
    MEMORY = "declined_memory"
    DISK = "declined_disk"
    THERMAL = "declined_thermal"
    UNHEALTHY = "unhealthy"


class SchedulerError(ValueError):
    pass


@dataclass(frozen=True)
class DeviceResponse:
    decision: DeviceDecision
    reason: DeclineReason | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def accepted(cls) -> "DeviceResponse":
        return cls(DeviceDecision.ACCEPTED)

    @classmethod
    def declined(
        cls, reason: DeclineReason, details: Mapping[str, Any] | None = None
    ) -> "DeviceResponse":
        return cls(DeviceDecision.DECLINED, reason, details or {})

    def __post_init__(self) -> None:
        if self.decision is DeviceDecision.ACCEPTED and self.reason is not None:
            raise ValueError("an accepted response cannot have a decline reason")
        if self.decision is DeviceDecision.DECLINED and self.reason is None:
            raise ValueError("a declined response requires a reason")


def dispatch(
    job: Mapping[str, Any],
    devices: Sequence[Device],
    attempt: Callable[[Device], DeviceResponse],
) -> dict[str, Any]:
    required = required_memory_gib(job)
    attempts: list[dict[str, Any]] = []

    for device in ordered_devices(job, devices):
        response = attempt(device)
        attempt_record = {
            "device": device.name,
            "label": device.label,
            "memory_gib": device.memory_gib,
            "decision": response.decision.value,
            "reason": response.reason.value if response.reason else None,
            "details": dict(response.details),
        }
        attempts.append(attempt_record)
        if response.decision is DeviceDecision.ACCEPTED:
            return _result(
                job,
                "accepted",
                required,
                attempts,
                selected_device=asdict(device),
                unavailable=_unavailable(devices, required),
            )

    return _result(
        job,
        "no_eligible_runner",
        required,
        attempts,
        unavailable=_unavailable(devices, required),
    )


def create_dispatch(
    job: Mapping[str, Any], devices: Sequence[Device]
) -> dict[str, Any]:
    required = required_memory_gib(job)
    required_disk = required_disk_gib(job)
    worker_job = dict(job)
    worker_job["required_memory_gib"] = required
    worker_job["required_disk_gib"] = required_disk
    candidates = [asdict(device) for device in ordered_devices(job, devices)]
    outcome = "dispatching" if candidates else "no_eligible_runner"
    return {
        "schema_version": 1,
        "kind": "device_dispatch",
        "job": worker_job,
        "required_memory_gib": required,
        "required_disk_gib": required_disk,
        "candidates": candidates,
        "unavailable": _unavailable(devices, required),
        "attempts": [],
        "outcome": outcome,
        "next_device": candidates[0] if candidates else None,
        "selected_device": None,
    }


def record_response(
    state: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_state(state)
    if state["outcome"] != "dispatching" or not state.get("next_device"):
        raise SchedulerError("dispatch is not awaiting a device response")
    if (
        response.get("schema_version") != 1
        or response.get("kind") != "device_job_result"
    ):
        raise SchedulerError("invalid device response")

    expected = state["next_device"]
    if response.get("job_id") != state["job"].get("id"):
        raise SchedulerError("device response job id does not match dispatch")
    if response.get("device") != expected.get("name"):
        raise SchedulerError("device response is not from the selected candidate")

    decision = response.get("decision")
    if decision not in {item.value for item in DeviceDecision}:
        raise SchedulerError("device response has an invalid decision")
    reason = response.get("reason")
    if decision == DeviceDecision.DECLINED.value and reason not in {
        item.value for item in DeclineReason
    }:
        raise SchedulerError("declined device response has an invalid reason")
    if decision == DeviceDecision.ACCEPTED.value and reason is not None:
        raise SchedulerError("accepted device response cannot have a decline reason")
    observed = response.get("observed", {})
    if not isinstance(observed, Mapping):
        raise SchedulerError("device response observations must be an object")

    attempt_record = {
        "device": expected["name"],
        "label": expected["label"],
        "memory_gib": expected["memory_gib"],
        "decision": decision,
        "reason": reason,
        "details": dict(observed),
    }
    updated = dict(state)
    attempts = [*state["attempts"], attempt_record]
    updated["attempts"] = attempts

    if decision == DeviceDecision.ACCEPTED.value:
        updated.update(
            {
                "outcome": "accepted",
                "next_device": None,
                "selected_device": dict(expected),
                "worker_outcome": response.get("outcome"),
            }
        )
        return updated

    tried = {item["label"] for item in attempts}
    remaining = [
        candidate
        for candidate in state["candidates"]
        if candidate["label"] not in tried
    ]
    updated["next_device"] = remaining[0] if remaining else None
    if not remaining:
        updated["outcome"] = "no_eligible_runner"
    return updated


def bot_result(state: Mapping[str, Any]) -> dict[str, Any]:
    _validate_state(state)
    if state["outcome"] != "no_eligible_runner":
        raise SchedulerError("dispatch has not exhausted its candidates")
    job = state["job"]
    return _result(
        job,
        "no_eligible_runner",
        state["required_memory_gib"],
        state["attempts"],
        unavailable=state["unavailable"],
    )


def _result(
    job: Mapping[str, Any],
    outcome: str,
    required: int,
    attempts: Sequence[Mapping[str, Any]],
    *,
    selected_device: Mapping[str, Any] | None = None,
    unavailable: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "component": str(job.get("component", "runner")),
        "model": job.get("model"),
        "profile": job.get("profile"),
        "job_id": str(job.get("id", "")),
        "outcome": outcome,
        "required_memory_gib": required,
        "required_disk_gib": required_disk_gib(job),
        "attempts": list(attempts),
        "unavailable": list(unavailable),
        "selected_device": dict(selected_device) if selected_device else None,
    }


def _unavailable(devices: Sequence[Device], required: int) -> list[dict[str, Any]]:
    unavailable: list[dict[str, Any]] = []
    for device in devices:
        reason = None
        if device.memory_gib < required:
            reason = DeclineReason.MEMORY.value
        elif not device.online or not device.healthy:
            reason = DeclineReason.UNHEALTHY.value
        elif device.busy:
            reason = DeclineReason.BUSY.value
        if reason:
            unavailable.append(
                {
                    "device": device.name,
                    "memory_gib": device.memory_gib,
                    "reason": reason,
                }
            )
    return unavailable


def _validate_state(state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != 1 or state.get("kind") != "device_dispatch":
        raise SchedulerError("invalid dispatch state")
    if not isinstance(state.get("job"), Mapping):
        raise SchedulerError("dispatch state has no job")
    for key in ("candidates", "unavailable", "attempts"):
        if not isinstance(state.get(key), list):
            raise SchedulerError(f"dispatch state {key} must be a list")
    if state.get("outcome") not in {
        "dispatching",
        "accepted",
        "no_eligible_runner",
    }:
        raise SchedulerError("dispatch state has an invalid outcome")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--job", type=Path, required=True)
    create_parser.add_argument("--devices", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--dispatch", type=Path, required=True)
    record_parser.add_argument("--response", type=Path, required=True)
    record_parser.add_argument("--output", type=Path, required=True)

    result_parser = subparsers.add_parser("bot-result")
    result_parser.add_argument("--dispatch", type=Path, required=True)
    result_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "create":
        job = _load_json(args.job)
        raw_devices = _load_json(args.devices)
        if not isinstance(job, Mapping) or not isinstance(raw_devices, list):
            raise SchedulerError("job must be an object and devices must be a list")
        devices = [Device(**device) for device in raw_devices]
        value = create_dispatch(job, devices)
    elif args.command == "record":
        value = record_response(_load_json(args.dispatch), _load_json(args.response))
    else:
        value = bot_result(_load_json(args.dispatch))
    _write_json(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
