import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import ci.device_lease as device_lease
from ci.device_lease import (
    ZERO_OID,
    DeviceLease,
    GhClient,
    GitHubApiError,
    GitHubRefLeaseStore,
    acquire_dispatch_lease,
    acquire_plan_leases,
    release_batch,
    release_job,
    retry_batch,
    write_batch,
)
from ci.runner_selection import Device

NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def test_gh_client_resolves_homebrew_outside_service_path(monkeypatch, tmp_path):
    executable = tmp_path / "gh"
    executable.touch(mode=0o755)
    monkeypatch.setattr(device_lease.shutil, "which", lambda name: None)
    monkeypatch.setattr(device_lease, "GH_EXECUTABLE_PATHS", (str(executable),))

    client = GhClient()

    assert client.executable == str(executable)


class FakeGitHubClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.refs = {}
        self.tags = {}

    def rest(self, endpoint, *, method="GET", body=None):
        with self.lock:
            if endpoint == "repos/org/repo":
                return {"node_id": "repository-node"}
            if endpoint == "repos/org/repo/git/commits" and method == "POST":
                oid = f"{len(self.tags) + 1:040x}"
                self.tags[oid] = dict(body)
                return {"sha": oid}
            ref_prefix = "repos/org/repo/git/ref/"
            if endpoint.startswith(ref_prefix):
                name = f"refs/{endpoint.removeprefix(ref_prefix)}"
                if name not in self.refs:
                    raise GitHubApiError("missing", "HTTP 404: Not Found")
                return {"object": {"sha": self.refs[name], "type": "tag"}}
            commit_prefix = "repos/org/repo/git/commits/"
            if endpoint.startswith(commit_prefix):
                oid = endpoint.removeprefix(commit_prefix)
                if oid not in self.tags:
                    return {"sha": oid, "tree": {"sha": "tree"}}
                return {"sha": oid, "message": self.tags[oid]["message"]}
        raise AssertionError((method, endpoint, body))

    def graphql(self, query, variables):
        updates = variables["input"]["refUpdates"]
        with self.lock:
            for update in updates:
                current = self.refs.get(update["name"], ZERO_OID)
                if current != update["beforeOid"]:
                    raise GitHubApiError("conflict", "beforeOid did not match")
            for update in updates:
                if update["afterOid"] == ZERO_OID:
                    self.refs.pop(update["name"], None)
                else:
                    self.refs[update["name"]] = update["afterOid"]
        return {"data": {"updateRefs": {"clientMutationId": None}}}


def acquire(store, attempt, device="mini-1", now=NOW):
    return store.acquire(
        attempt_id=attempt,
        device=device,
        label=f"device-{device}",
        head_sha="head",
        target_sha="target",
        run_url=f"https://example.com/{attempt}",
        ttl_seconds=300,
        now=now,
    )


def dispatch():
    candidates = [
        {
            "name": "mini-1",
            "label": "device-mini-1",
            "memory_gib": 16,
            "online": True,
            "busy": False,
            "healthy": True,
        },
        {
            "name": "mini-2",
            "label": "device-mini-2",
            "memory_gib": 16,
            "online": True,
            "busy": False,
            "healthy": True,
        },
        {
            "name": "m5",
            "label": "device-m5",
            "memory_gib": 128,
            "online": True,
            "busy": False,
            "healthy": True,
        },
    ]
    return {
        "schema_version": 1,
        "kind": "device_dispatch",
        "job": {},
        "required_memory_gib": 8,
        "required_disk_gib": 4,
        "candidates": candidates,
        "unavailable": [],
        "attempts": [],
        "outcome": "dispatching",
        "next_device": candidates[0],
        "selected_device": None,
    }


def test_atomic_contention_allows_exactly_one_owner():
    client = FakeGitHubClient()

    def claim(index):
        return acquire(GitHubRefLeaseStore("org/repo", client), str(index))

    with ThreadPoolExecutor(max_workers=20) as executor:
        leases = list(executor.map(claim, range(100)))

    winners = [lease for lease in leases if lease is not None]
    assert len(winners) == 1
    assert GitHubRefLeaseStore("org/repo", client).get("mini-1") == winners[0]


def test_concurrent_dispatches_spread_once_then_report_capacity():
    client = FakeGitHubClient()

    def schedule(index):
        return acquire_dispatch_lease(
            dispatch(),
            GitHubRefLeaseStore("org/repo", client),
            attempt_id=str(index),
            head_sha="head",
            target_sha="target",
            run_url=f"run-{index}",
            ttl_seconds=300,
            now=NOW,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(schedule, range(4)))

    leases = [lease for _, lease in results if lease is not None]
    exhausted = [state for state, lease in results if lease is None]
    assert {lease.device for lease in leases} == {"mini-1", "mini-2", "m5"}
    assert len(exhausted) == 1
    assert exhausted[0]["outcome"] == "no_eligible_runner"


def test_dispatch_skips_leased_device_and_uses_next_smallest():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    assert acquire(store, "existing") is not None

    updated, lease = acquire_dispatch_lease(
        dispatch(),
        store,
        attempt_id="new",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    assert lease is not None
    assert lease.device == "mini-2"
    assert updated["next_device"]["name"] == "mini-2"
    assert updated["unavailable"] == [
        {
            "device": "mini-1",
            "memory_gib": 16,
            "reason": "leased",
            "attempt_id": "existing",
            "expires_at": "2026-08-30T00:05:00Z",
        }
    ]


def test_exhausted_leases_report_no_capacity():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    for device in ("mini-1", "mini-2", "m5"):
        assert acquire(store, f"owner-{device}", device) is not None

    updated, lease = acquire_dispatch_lease(
        dispatch(),
        store,
        attempt_id="waiting",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    assert lease is None
    assert updated["outcome"] == "no_eligible_runner"
    assert updated["next_device"] is None
    assert {item["reason"] for item in updated["unavailable"]} == {"leased"}


def test_stale_owner_cannot_release_replacement_lease():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    old = acquire(store, "old")
    assert old is not None
    new = acquire(store, "new", now=NOW + timedelta(minutes=6))
    assert new is not None

    assert store.release("mini-1", "old") is False
    assert store.get("mini-1") == new
    assert store.release("mini-1", "new") is True
    assert store.get("mini-1") is None


def test_heartbeat_extends_only_the_current_owner():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    original = acquire(store, "owner")
    assert original is not None

    renewed = store.heartbeat("mini-1", "owner", 300, now=NOW + timedelta(minutes=4))

    assert renewed.expires_at == "2026-08-30T00:09:00Z"
    assert renewed.token != original.token
    assert store.get("mini-1") == renewed


def test_lease_tag_contains_auditable_payload():
    client = FakeGitHubClient()
    lease = acquire(GitHubRefLeaseStore("org/repo", client), "123")
    assert lease is not None

    payload = json.loads(client.tags[lease.token]["message"])
    assert payload["attempt_id"] == "123"
    assert payload["device"] == "mini-1"
    assert payload["head_sha"] == "head"
    assert "token" not in payload


def model_work(model, weight_bytes):
    from ci.components.model_path import resource_requirements

    required_memory, required_disk = resource_requirements(
        {"weight": {"bytes": weight_bytes}}
    )
    return {
        "id": f"model_path:{model}",
        "work_type": "ModelPath",
        "component": "model_path",
        "subject": model,
        "model": model,
        "phases": ["synthetic", "hf_checkpoint"],
        "synthetic": {"adapter": model, "profile": "dense_vlm"},
        "hf_checkpoint": {"weight": {"bytes": weight_bytes}},
        "required_memory_gib": required_memory,
        "required_disk_gib": required_disk,
    }


def cache_work():
    return {
        "id": "kv_cache_change:dense",
        "work_type": "KVCacheChange",
        "component": "kv_cache_change",
        "profile": "dense",
        "phases": ["kv_cache_contract"],
        "required_memory_gib": 8,
        "required_disk_gib": 2,
    }


def test_ten_back_to_back_cache_prs_fan_out_then_report_capacity():
    client = FakeGitHubClient()
    devices = [
        Device("mini-1", "device-mini-1", 16),
        Device("mini-2", "device-mini-2", 16),
        Device("m5", "device-m5", 128),
    ]
    batches = [
        acquire_plan_leases(
            {"jobs": [cache_work()]},
            devices,
            GitHubRefLeaseStore("org/repo", client),
            attempt_id=f"pr-{index}",
            head_sha=f"head-{index}",
            target_sha="target",
            run_url=f"run-{index}",
            ttl_seconds=300,
            now=NOW,
        )
        for index in range(10)
    ]

    assigned = [
        batch["items"][0]["lease"]["device"]
        for batch in batches
        if batch["items"][0]["lease"] is not None
    ]
    waiting = [
        batch["items"][0] for batch in batches if batch["items"][0]["lease"] is None
    ]

    assert assigned == ["mini-1", "mini-2", "m5"]
    assert len(waiting) == 7
    assert all(item["dispatch"]["outcome"] == "no_eligible_runner" for item in waiting)
    assert all(item["work"]["profile"] == "dense" for item in batches[0]["items"])


def test_cache_contract_decline_retries_larger_device_with_profile_intact():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    devices = [
        Device("mini", "device-mini", 16),
        Device("m5", "device-m5", 128),
    ]
    first = acquire_plan_leases(
        {"jobs": [cache_work()]},
        devices,
        store,
        attempt_id="cache-attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )
    key = first["items"][0]["key"]

    second = retry_batch(
        first,
        {
            key: {
                "decision": "declined",
                "reason": "declined_disk",
                "observed": {"available_disk_gib": 1},
            }
        },
        devices,
        store,
        attempt_id="cache-attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    item = second["items"][0]
    assert item["execute"] is True
    assert item["work"]["profile"] == "dense"
    assert item["lease"]["device"] == "m5"
    assert item["dispatch"]["attempts"] == [
        {
            "device": "mini",
            "label": "device-mini",
            "memory_gib": 16,
            "decision": "declined",
            "reason": "declined_disk",
            "details": {"available_disk_gib": 1},
        }
    ]


def test_batch_assigns_largest_work_first_then_uses_smallest_fit(tmp_path):
    plan = {
        "jobs": [
            model_work("small-a", 1 * 2**30),
            model_work("large", 40 * 2**30),
            model_work("small-b", 1 * 2**30),
        ]
    }
    devices = [
        Device("mini-1", "device-mini-1", 16),
        Device("mini-2", "device-mini-2", 16),
        Device("m5", "device-m5", 128),
    ]

    batch = acquire_plan_leases(
        plan,
        devices,
        GitHubRefLeaseStore("org/repo", FakeGitHubClient()),
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )
    assignments = {
        item["work"]["model"]: item["lease"]["device"] for item in batch["items"]
    }
    matrix = write_batch(tmp_path, batch)

    assert assignments == {
        "large": "m5",
        "small-a": "mini-1",
        "small-b": "mini-2",
    }
    assert len(matrix["include"]) == 3
    assert all(item["work"]["required_memory_gib"] > 0 for item in batch["items"])
    assert all(item["work"]["required_disk_gib"] > 0 for item in batch["items"])
    assert all(item["lease"]["release_on_job_end"] is False for item in batch["items"])


def test_queued_job_release_defers_to_attempt_cleanup():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    batch = acquire_plan_leases(
        {"jobs": [model_work("one", 1), model_work("two", 1)]},
        [Device("mini", "device-mini", 16)],
        store,
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )
    lease = DeviceLease(**batch["items"][0]["lease"])

    assert release_job(lease, store) is True
    assert store.get("mini") is not None
    assert release_batch(batch, store) == {"mini": True}
    assert store.get("mini") is None


def test_batch_queues_additional_work_on_attempt_owned_devices():
    plan = {
        "jobs": [
            model_work("one", 1),
            model_work("two", 1),
            model_work("three", 1),
            model_work("four", 1),
        ]
    }
    devices = [
        Device("mini-1", "device-mini-1", 16),
        Device("mini-2", "device-mini-2", 16),
        Device("m5", "device-m5", 128),
    ]

    batch = acquire_plan_leases(
        plan,
        devices,
        GitHubRefLeaseStore("org/repo", FakeGitHubClient()),
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    assert len([item for item in batch["items"] if item["lease"]]) == 4
    assignments = [item["lease"]["device"] for item in batch["items"]]
    assert assignments.count("mini-1") == 2
    assert assignments.count("mini-2") == 1
    assert assignments.count("m5") == 1


def test_declined_small_runner_is_retried_on_next_larger_device():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    devices = [
        Device("mini", "device-mini", 16),
        Device("m5", "device-m5", 128),
    ]
    first = acquire_plan_leases(
        {"jobs": [model_work("qwen", 1)]},
        devices,
        store,
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )
    key = first["items"][0]["key"]
    results = {
        key: {
            "decision": "declined",
            "reason": "declined_disk",
            "observed": {"available_disk_gib": 1},
        }
    }

    second = retry_batch(
        first,
        results,
        devices,
        store,
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    item = second["items"][0]
    assert item["execute"] is True
    assert item["lease"]["device"] == "m5"
    assert item["dispatch"]["attempts"][0]["device"] == "mini"
    assert item["dispatch"]["attempts"][0]["reason"] == "declined_disk"


def test_accepted_work_is_not_scheduled_again():
    client = FakeGitHubClient()
    store = GitHubRefLeaseStore("org/repo", client)
    devices = [Device("mini", "device-mini", 16)]
    first = acquire_plan_leases(
        {"jobs": [model_work("qwen", 1)]},
        devices,
        store,
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )
    key = first["items"][0]["key"]

    second = retry_batch(
        first,
        {key: {"decision": "accepted", "outcome": "passed"}},
        devices,
        store,
        attempt_id="attempt",
        head_sha="head",
        target_sha="target",
        run_url="run",
        ttl_seconds=300,
        now=NOW,
    )

    assert second["items"][0]["execute"] is False
