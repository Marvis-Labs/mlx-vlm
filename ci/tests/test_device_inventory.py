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


def job():
    return {
        "id": "model_path:qwen2_vl",
        "work_type": "ModelPath",
        "component": "model_path",
        "model": "qwen2_vl",
        "phases": ["synthetic", "hf_checkpoint"],
        "hf_checkpoint": {"weight": {"bytes": 2**30}},
    }


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
    second = job()
    second["id"] = "model_path:second"
    second["model"] = "second"
    with pytest.raises(DeviceInventoryError, match="exactly one"):
        checkpoint_job({"jobs": [job(), second]})


def test_configured_devices_marks_active_runner_busy():
    devices = configured_devices(
        [{"name": "mini", "label": "device-mini", "memory_gib": 16}],
        {"mini"},
    )
    assert devices[0].busy is True


def test_configured_devices_uses_live_online_state():
    devices = configured_devices(
        [
            {"name": "mini", "label": "device-mini", "memory_gib": 16},
            {"name": "m5", "label": "device-m5", "memory_gib": 128},
            {"name": "missing", "label": "device-missing", "memory_gib": 64},
        ],
        set(),
        {
            "runners": [
                {"name": "mini", "status": "online", "busy": False},
                {"name": "m5", "status": "offline", "busy": False},
            ]
        },
    )

    assert devices[0].online is True
    assert devices[1].online is False
    assert devices[2].online is False


def test_cli_selects_smallest_idle_device(tmp_path):
    plan = tmp_path / "plan.json"
    devices = tmp_path / "devices.json"
    busy = tmp_path / "busy.json"
    dispatch = tmp_path / "dispatch.json"
    selected_job = tmp_path / "job.json"
    plan.write_text(json.dumps({"jobs": [job()]}))
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
            ]
        )
        == 0
    )
    assert json.loads(dispatch.read_text())["next_device"]["name"] == "mini"
