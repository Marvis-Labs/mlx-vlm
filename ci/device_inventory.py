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
    payload: Sequence[Mapping[str, Any]], busy_names: set[str]
) -> list[Device]:
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
        devices.append(
            Device(
                name=name,
                label=label,
                memory_gib=memory_gib,
                busy=name in busy_names,
            )
        )
    return devices


def checkpoint_job(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise DeviceInventoryError("approved plan must contain a jobs list")
    selected = [
        job
        for job in jobs
        if isinstance(job, Mapping)
        and job.get("component") in {"model_path", "new_model_path"}
        and job.get("mode") == "hf_checkpoint"
    ]
    if len(selected) != 1:
        raise DeviceInventoryError(
            "this workflow currently requires exactly one checkpoint model job"
        )
    return selected[0]


def write_github_output(path: Path, has_device: bool, label: str | None) -> None:
    runs_on = json.dumps(["self-hosted", label]) if label else "[]"
    with path.open("a") as stream:
        stream.write(f"has_device={'true' if has_device else 'false'}\n")
        stream.write(f"runs_on={runs_on}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--busy", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
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
    next_device = state.get("next_device")
    label = next_device.get("label") if isinstance(next_device, Mapping) else None
    write_github_output(args.github_output, label is not None, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
