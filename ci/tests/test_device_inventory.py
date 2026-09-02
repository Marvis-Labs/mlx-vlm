import pytest

from ci.device_inventory import (
    DeviceInventoryError,
    configured_devices,
    devices_from_github,
    work_items,
)


def runner(name, memory, *, status="online", busy=False):
    return {
        "name": name,
        "status": status,
        "busy": busy,
        "labels": [
            {"name": "self-hosted"},
            {"name": "apple-silicon"},
            {"name": "mlx-ci-sandbox-v1"},
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


def test_github_inventory_rejects_runner_without_security_profile():
    untrusted = runner("legacy", 128)
    untrusted["labels"] = [
        label for label in untrusted["labels"] if label["name"] != "mlx-ci-sandbox-v1"
    ]

    assert devices_from_github({"runners": [untrusted]}) == []


def test_inventory_accepts_multiple_independent_work_items():
    second = job()
    second["id"] = "model_path:second"
    second["model"] = "second"

    selected = work_items({"jobs": [job(), second]})

    assert [item["model"] for item in selected] == ["qwen2_vl", "second"]


def test_inventory_rejects_unsupported_work():
    unsupported = job()
    unsupported["work_type"] = "ComponentPath"

    with pytest.raises(DeviceInventoryError, match="unsupported work"):
        work_items({"jobs": [unsupported]})


def test_inventory_accepts_kv_cache_change_work():
    cache = {
        "id": "kv_cache_change:dense",
        "work_type": "KVCacheChange",
        "component": "kv_cache_change",
        "profile": "dense",
        "phases": ["kv_cache_contract"],
        "required_memory_gib": 8,
        "required_disk_gib": 2,
    }

    assert work_items({"jobs": [cache]}) == [cache]


def test_configured_devices_marks_active_runner_busy():
    devices = configured_devices(
        [
            {
                "name": "mini",
                "label": "device-mini",
                "memory_gib": 16,
                "security_profile": "mlx-ci-sandbox-v1",
            }
        ],
        {"mini"},
    )
    assert devices[0].busy is True


def test_configured_devices_rejects_legacy_runner_profile():
    with pytest.raises(DeviceInventoryError, match="configured device is invalid"):
        configured_devices(
            [{"name": "legacy", "label": "device-legacy", "memory_gib": 128}],
            set(),
        )


def test_configured_devices_uses_live_online_state():
    devices = configured_devices(
        [
            {
                "name": "mini",
                "label": "device-mini",
                "memory_gib": 16,
                "security_profile": "mlx-ci-sandbox-v1",
            },
            {
                "name": "m5",
                "label": "device-m5",
                "memory_gib": 128,
                "security_profile": "mlx-ci-sandbox-v1",
            },
            {
                "name": "missing",
                "label": "device-missing",
                "memory_gib": 64,
                "security_profile": "mlx-ci-sandbox-v1",
            },
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
