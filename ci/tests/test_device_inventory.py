import json

import pytest

from ci.device_inventory import (
    DeviceInventoryError,
    checkpoint_job,
    configured_devices,
    devices_from_github,
    main,
)


def runner(name, memory, *, status="online", busy=False):
    return {
        "name": name,
        "status": status,
        "busy": busy,
        "labels": [
            {"name": "self-hosted"},
            {"name": "apple-silicon"},
            {"name": f"device-{name}"},
            {"name": f"memory-{memory}gb"},
        ],
    }


def job(mode):
    value = {
        "id": f"model_path:qwen2_vl:{mode}",
        "component": "model_path",
        "model": "qwen2_vl",
        "mode": mode,
    }
    if mode == "hf_checkpoint":
        value["hf_checkpoint"] = {"weight": {"bytes": 2**30}}
    return value


def test_github_inventory_preserves_order_and_runner_state():
    payload = {
        "runners": [
            runner("m5", 128),
            runner("mini-1", 16),
            runner("mini-2", 16, busy=True),
        ]
    }

    devices = devices_from_github(payload)

    assert [device.name for device in devices] == ["m5", "mini-1", "mini-2"]
    assert devices[-1].busy is True


def test_checkpoint_job_requires_exactly_one_model():
    with pytest.raises(DeviceInventoryError, match="exactly one"):
        checkpoint_job({"jobs": [job("hf_checkpoint"), job("hf_checkpoint")]})


def test_configured_devices_marks_active_runner_busy():
    devices = configured_devices(
        [{"name": "mini", "label": "device-mini", "memory_gib": 16}],
        {"mini"},
    )
    assert devices[0].busy is True


def test_cli_selects_smallest_idle_device(tmp_path):
    plan = tmp_path / "plan.json"
    devices = tmp_path / "devices.json"
    busy = tmp_path / "busy.json"
    dispatch = tmp_path / "dispatch.json"
    selected_job = tmp_path / "job.json"
    output = tmp_path / "github-output"
    plan.write_text(json.dumps({"jobs": [job("synthetic"), job("hf_checkpoint")]}))
    devices.write_text(
        json.dumps(
            [
                {"name": "m5", "label": "device-m5", "memory_gib": 128},
                {"name": "mini", "label": "device-mini", "memory_gib": 16},
            ]
        )
    )
    busy.write_text("[]")

    assert (
        main(
            [
                "--plan",
                str(plan),
                "--devices",
                str(devices),
                "--busy",
                str(busy),
                "--dispatch",
                str(dispatch),
                "--job",
                str(selected_job),
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(dispatch.read_text())["next_device"]["name"] == "mini"
    assert 'runs_on=["self-hosted", "device-mini"]' in output.read_text()
