from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.runner_selection import Device
from ci.scheduler import create_dispatch


class DeviceInventoryError(ValueError):
    pass


def devices_from_github(payload: Mapping[str, Any]) -> list[Device]:
    runners = payload.get("runners")
    if not isinstance(runners, list):
        raise DeviceInventoryError("runner inventory must contain a runners list")

    devices: list[Device] = []
    for runner in runners:
        if not isinstance(runner, Mapping):
            continue
        labels = runner.get("labels", [])
        names = [
            str(label.get("name"))
            for label in labels
            if isinstance(label, Mapping) and label.get("name")
        ]
        if "apple-silicon" not in names:
            continue
        device_labels = [name for name in names if name.startswith("device-")]
        memory_values = [
            int(name.removeprefix("memory-").removesuffix("gb"))
            for name in names
            if name.startswith("memory-")
            and name.endswith("gb")
            and name.removeprefix("memory-").removesuffix("gb").isdigit()
        ]
        if len(device_labels) != 1 or not memory_values:
            continue
        devices.append(
            Device(
                name=str(runner.get("name", device_labels[0])),
                label=device_labels[0],
                memory_gib=max(memory_values),
                online=runner.get("status") == "online",
                busy=bool(runner.get("busy", False)),
            )
        )
    return devices


def configured_devices(
    payload: Sequence[Mapping[str, Any]],
    busy_names: set[str],
    live_payload: Mapping[str, Any] | None = None,
) -> list[Device]:
    live_runners: dict[str, Mapping[str, Any]] | None = None
    if live_payload is not None:
        runners = live_payload.get("runners")
        if not isinstance(runners, list):
            raise DeviceInventoryError(
                "live runner inventory must contain a runners list"
            )
        live_runners = {
            str(runner.get("name")): runner
            for runner in runners
            if isinstance(runner, Mapping) and runner.get("name")
        }
    devices: list[Device] = []
    for item in payload:
        name = item.get("name")
        label = item.get("label")
        memory_gib = item.get("memory_gib")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(label, str)
            or not label.startswith("device-")
            or not isinstance(memory_gib, int)
            or memory_gib <= 0
        ):
            raise DeviceInventoryError("configured device is invalid")
        live = live_runners.get(name) if live_runners is not None else None
        devices.append(
            Device(
                name=name,
                label=label,
                memory_gib=memory_gib,
                online=(
                    (live is not None and live.get("status") == "online")
                    if live_runners is not None
                    else True
                ),
                busy=(
                    (name in busy_names or bool(live.get("busy", False)))
                    if live is not None
                    else name in busy_names
                ),
            )
        )
    return devices


def model_path_work_items(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise DeviceInventoryError("approved plan must contain a jobs list")
    selected = [
        job
        for job in jobs
        if isinstance(job, Mapping)
        and job.get("component") == "model_path"
        and job.get("work_type") == "ModelPath"
    ]
    if len(selected) != len(jobs):
        raise DeviceInventoryError("approved plan contains unsupported work")
    identifiers = [job.get("id") for job in selected]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise DeviceInventoryError("every ModelPath work item requires an id")
    if len(identifiers) != len(set(identifiers)):
        raise DeviceInventoryError("ModelPath work item ids must be unique")
    return selected


def checkpoint_job(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = model_path_work_items(plan)
    if len(selected) != 1:
        raise DeviceInventoryError("exactly one ModelPath work item is required")
    return selected[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--busy", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    device_config = json.loads(args.devices.read_text())
    busy = json.loads(args.busy.read_text())
    if not isinstance(plan, Mapping) or not isinstance(device_config, list):
        raise DeviceInventoryError("plan must be an object and devices must be a list")
    if not isinstance(busy, list) or any(not isinstance(name, str) for name in busy):
        raise DeviceInventoryError("busy runner inventory must be a list of names")

    state = create_dispatch(
        checkpoint_job(plan), configured_devices(device_config, set(busy))
    )
    args.dispatch.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    args.job.write_text(json.dumps(state["job"], indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
