import json

import pytest

from ci.runner_selection import Device
from ci.scheduler import (
    DeclineReason,
    DeviceResponse,
    SchedulerError,
    bot_result,
    create_dispatch,
    dispatch,
    main,
    record_response,
)


def job(weight_gib=1):
    return {
        "id": "model_path:qwen2_vl:hf_checkpoint",
        "component": "model_path",
        "model": "qwen2_vl",
        "mode": "hf_checkpoint",
        "hf_checkpoint": {"weight": {"bytes": int(weight_gib * 2**30)}},
    }


def test_declines_try_same_tier_then_escalate_to_bigger_device():
    devices = [
        Device("m5", "device-m5", 128),
        Device("mini-1", "device-mini-1", 16),
        Device("mid", "device-mid", 64),
        Device("mini-2", "device-mini-2", 16),
    ]
    responses = {
        "mini-1": DeviceResponse.declined(DeclineReason.BUSY),
        "mini-2": DeviceResponse.declined(DeclineReason.DISK),
        "mid": DeviceResponse.accepted(),
    }

    result = dispatch(job(), devices, lambda device: responses[device.name])

    assert result["outcome"] == "accepted"
    assert result["selected_device"]["name"] == "mid"
    assert [attempt["device"] for attempt in result["attempts"]] == [
        "mini-1",
        "mini-2",
        "mid",
    ]


def test_all_declines_return_structured_no_capacity_result():
    devices = [
        Device("mini", "device-mini", 16),
        Device("m5", "device-m5", 128),
    ]

    result = dispatch(
        job(),
        devices,
        lambda device: DeviceResponse.declined(
            DeclineReason.MEMORY, {"available_memory_gib": device.memory_gib - 1}
        ),
    )

    assert result["outcome"] == "no_eligible_runner"
    assert result["required_memory_gib"] == 8
    assert [attempt["device"] for attempt in result["attempts"]] == ["mini", "m5"]
    assert result["selected_device"] is None


def test_initially_unavailable_devices_are_reported_without_attempting_them():
    devices = [
        Device("small", "device-small", 16),
        Device("busy", "device-busy", 128, busy=True),
        Device("offline", "device-offline", 128, online=False),
    ]

    result = dispatch(
        job(40),
        devices,
        lambda device: DeviceResponse.accepted(),
    )

    assert result["outcome"] == "no_eligible_runner"
    assert result["attempts"] == []
    assert {item["reason"] for item in result["unavailable"]} == {
        "declined_memory",
        "declined_busy",
        "unhealthy",
    }


def response(device, decision, reason=None, outcome="declined"):
    return {
        "schema_version": 1,
        "kind": "device_job_result",
        "job_id": "model_path:qwen2_vl:hf_checkpoint",
        "device": device,
        "decision": decision,
        "outcome": outcome,
        "reason": reason,
        "observed": {"memory_gib": 16, "available_disk_gib": 10},
        "exit_code": 0,
    }


def test_serializable_dispatch_advances_after_decline():
    devices = [
        Device("m5", "device-m5", 128),
        Device("mini", "device-mini", 16),
    ]
    state = create_dispatch(job(), devices)

    assert state["next_device"]["name"] == "mini"
    assert state["job"]["required_memory_gib"] == 8
    assert state["job"]["required_disk_gib"] == 4

    state = record_response(state, response("mini", "declined", "declined_memory"))

    assert state["outcome"] == "dispatching"
    assert state["next_device"]["name"] == "m5"

    state = record_response(state, response("m5", "accepted", outcome="passed"))

    assert state["outcome"] == "accepted"
    assert state["selected_device"]["name"] == "m5"
    assert state["worker_outcome"] == "passed"


def test_serializable_dispatch_exhaustion_becomes_bot_result():
    state = create_dispatch(job(), [Device("mini", "device-mini", 16)])
    state = record_response(state, response("mini", "declined", "declined_disk"))

    result = bot_result(state)

    assert state["outcome"] == "no_eligible_runner"
    assert result["outcome"] == "no_eligible_runner"
    assert result["attempts"][0]["reason"] == "declined_disk"


def test_serializable_dispatch_rejects_response_from_wrong_device():
    state = create_dispatch(job(), [Device("mini", "device-mini", 16)])

    with pytest.raises(SchedulerError, match="selected candidate"):
        record_response(state, response("different", "declined", "declined_busy"))


def test_serializable_dispatch_rejects_invalid_device_result_schema():
    state = create_dispatch(job(), [Device("mini", "device-mini", 16)])
    invalid = response("mini", "declined", "declined_busy")
    invalid["schema_version"] = 2

    with pytest.raises(SchedulerError, match="invalid device response"):
        record_response(state, invalid)


def test_scheduler_cli_round_trip(tmp_path):
    job_path = tmp_path / "job.json"
    devices_path = tmp_path / "devices.json"
    dispatch_path = tmp_path / "dispatch.json"
    response_path = tmp_path / "response.json"
    exhausted_path = tmp_path / "exhausted.json"
    bot_result_path = tmp_path / "bot-result.json"
    job_path.write_text(json.dumps(job()))
    devices_path.write_text(
        json.dumps(
            [
                {
                    "name": "mini",
                    "label": "device-mini",
                    "memory_gib": 16,
                }
            ]
        )
    )
    response_path.write_text(json.dumps(response("mini", "declined", "declined_busy")))

    assert (
        main(
            [
                "create",
                "--job",
                str(job_path),
                "--devices",
                str(devices_path),
                "--output",
                str(dispatch_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "record",
                "--dispatch",
                str(dispatch_path),
                "--response",
                str(response_path),
                "--output",
                str(exhausted_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "bot-result",
                "--dispatch",
                str(exhausted_path),
                "--output",
                str(bot_result_path),
            ]
        )
        == 0
    )
    assert json.loads(bot_result_path.read_text())["outcome"] == "no_eligible_runner"
