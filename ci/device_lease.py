from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ci.device_inventory import configured_devices, work_items
from ci.runner_selection import Device, required_memory_gib
from ci.scheduler import create_dispatch

ZERO_OID = "0" * 40
REF_PREFIX = "refs/tags/ci-device-lease"
UPDATE_REFS_MUTATION = """
mutation UpdateLeaseRefs($input: UpdateRefsInput!) {
  updateRefs(input: $input) { clientMutationId }
}
"""


class DeviceLeaseError(RuntimeError):
    pass


class GitHubApiError(DeviceLeaseError):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


@dataclass(frozen=True)
class DeviceLease:
    schema_version: int
    attempt_id: str
    device: str
    label: str
    head_sha: str
    target_sha: str
    run_url: str
    acquired_at: str
    heartbeat_at: str
    expires_at: str
    generation: str
    token: str = ""
    release_on_job_end: bool = True

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token")
        return value

    def expired(self, now: datetime) -> bool:
        return _parse_time(self.expires_at) <= now


class GitHubClient(Protocol):
    def rest(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def graphql(
        self, query: str, variables: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class GhClient:
    def rest(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        command = ["gh", "api"]
        if method != "GET":
            command.extend(["--method", method])
        command.append(endpoint)
        return self._run(command, body)

    def graphql(self, query: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._run(
            ["gh", "api", "graphql"], {"query": query, "variables": variables}
        )

    def _run(
        self, command: list[str], body: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        if body is not None:
            command.extend(["--input", "-"])
        completed = subprocess.run(
            command,
            input=json.dumps(body) if body is not None else None,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise GitHubApiError("GitHub API request failed", completed.stderr.strip())
        if not completed.stdout.strip():
            return {}
        value = json.loads(completed.stdout)
        if not isinstance(value, Mapping):
            raise GitHubApiError("GitHub API response must be an object")
        return value


class GitHubRefLeaseStore:
    def __init__(self, repository: str, client: GitHubClient | None = None):
        if not re.fullmatch(r"[^/]+/[^/]+", repository):
            raise DeviceLeaseError("repository must use owner/name format")
        self.repository = repository
        self.client = client or GhClient()
        self._repository_node_id: str | None = None

    def acquire(
        self,
        *,
        attempt_id: str,
        device: str,
        label: str,
        head_sha: str,
        target_sha: str,
        run_url: str,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> DeviceLease | None:
        instant = _utc(now)
        current = self.get(device)
        if current is not None and not current.expired(instant):
            return None
        candidate = _new_lease(
            attempt_id=attempt_id,
            device=device,
            label=label,
            head_sha=head_sha,
            target_sha=target_sha,
            run_url=run_url,
            ttl_seconds=ttl_seconds,
            now=instant,
        )
        commit_oid = self._create_payload_commit(candidate)
        before_oid = current.token if current is not None else ZERO_OID
        try:
            self._update_ref(device, before_oid, commit_oid)
        except GitHubApiError:
            refreshed = self.get(device)
            if refreshed is not None and refreshed.token != before_oid:
                return None
            raise
        return replace(candidate, token=commit_oid)

    def get(self, device: str) -> DeviceLease | None:
        ref = _ref_name(device)
        try:
            value = self.client.rest(
                f"repos/{self.repository}/git/ref/{ref.removeprefix('refs/')}"
            )
        except GitHubApiError as error:
            if "404" in error.stderr or "Not Found" in error.stderr:
                return None
            raise
        target = value.get("object")
        if not isinstance(target, Mapping) or not isinstance(target.get("sha"), str):
            raise DeviceLeaseError("lease ref has no target object")
        token = str(target["sha"])
        commit = self.client.rest(f"repos/{self.repository}/git/commits/{token}")
        message = commit.get("message")
        if not isinstance(message, str):
            raise DeviceLeaseError("lease tag has no message")
        payload = json.loads(message)
        if not isinstance(payload, Mapping):
            raise DeviceLeaseError("lease payload must be an object")
        lease = DeviceLease(**payload, token=token)
        if lease.schema_version != 1 or lease.device != device:
            raise DeviceLeaseError("lease payload does not match its device ref")
        return lease

    def heartbeat(
        self,
        device: str,
        attempt_id: str,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> DeviceLease:
        current = self.get(device)
        if current is None or current.attempt_id != attempt_id:
            raise DeviceLeaseError("device lease is no longer owned by this attempt")
        instant = _utc(now)
        updated = replace(
            current,
            heartbeat_at=_format_time(instant),
            expires_at=_format_time(instant + timedelta(seconds=ttl_seconds)),
            generation=uuid.uuid4().hex,
            token="",
        )
        commit_oid = self._create_payload_commit(updated)
        self._update_ref(device, current.token, commit_oid)
        return replace(updated, token=commit_oid)

    def release(self, device: str, attempt_id: str) -> bool:
        current = self.get(device)
        if current is None:
            return True
        if current.attempt_id != attempt_id:
            return False
        try:
            self._update_ref(device, current.token, ZERO_OID)
        except GitHubApiError:
            return False
        return True

    def _create_payload_commit(self, lease: DeviceLease) -> str:
        target = self.client.rest(
            f"repos/{self.repository}/git/commits/{lease.target_sha}"
        )
        tree = target.get("tree")
        if not isinstance(tree, Mapping) or not isinstance(tree.get("sha"), str):
            raise DeviceLeaseError("lease target commit has no tree")
        value = self.client.rest(
            f"repos/{self.repository}/git/commits",
            method="POST",
            body={
                "message": json.dumps(
                    lease.payload(), sort_keys=True, separators=(",", ":")
                ),
                "tree": tree["sha"],
                "parents": [lease.target_sha],
            },
        )
        oid = value.get("sha")
        if not isinstance(oid, str) or not oid:
            raise DeviceLeaseError("GitHub did not return a lease commit object ID")
        return oid

    def _update_ref(self, device: str, before_oid: str, after_oid: str) -> None:
        self.client.graphql(
            UPDATE_REFS_MUTATION,
            {
                "input": {
                    "repositoryId": self._repository_id(),
                    "refUpdates": [
                        {
                            "name": _ref_name(device),
                            "beforeOid": before_oid,
                            "afterOid": after_oid,
                            "force": True,
                        }
                    ],
                }
            },
        )

    def _repository_id(self) -> str:
        if self._repository_node_id is None:
            value = self.client.rest(f"repos/{self.repository}")
            node_id = value.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise DeviceLeaseError("repository has no GraphQL node ID")
            self._repository_node_id = node_id
        return self._repository_node_id


def acquire_dispatch_lease(
    dispatch: Mapping[str, Any],
    store: GitHubRefLeaseStore,
    *,
    attempt_id: str,
    head_sha: str,
    target_sha: str,
    run_url: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> tuple[dict[str, Any], DeviceLease | None]:
    candidates = dispatch.get("candidates")
    if not isinstance(candidates, list):
        raise DeviceLeaseError("dispatch has no candidate list")
    updated = dict(dispatch)
    unavailable = list(dispatch.get("unavailable", []))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise DeviceLeaseError("dispatch candidate must be an object")
        device = str(candidate.get("name", ""))
        label = str(candidate.get("label", ""))
        lease = store.acquire(
            attempt_id=attempt_id,
            device=device,
            label=label,
            head_sha=head_sha,
            target_sha=target_sha,
            run_url=run_url,
            ttl_seconds=ttl_seconds,
            now=now,
        )
        if lease is not None:
            updated["next_device"] = dict(candidate)
            updated["lease"] = asdict(lease)
            updated["outcome"] = "dispatching"
            updated["unavailable"] = unavailable
            return updated, lease
        owner = store.get(device)
        record = {
            "device": device,
            "memory_gib": candidate.get("memory_gib"),
            "reason": "leased",
        }
        if owner is not None:
            record.update(
                {"attempt_id": owner.attempt_id, "expires_at": owner.expires_at}
            )
        unavailable.append(record)
    updated["next_device"] = None
    updated["lease"] = None
    updated["outcome"] = "no_eligible_runner"
    updated["unavailable"] = unavailable
    return updated, None


def write_github_output(path: Path, lease: DeviceLease | None) -> None:
    label = lease.label if lease is not None else None
    runs_on = json.dumps(["self-hosted", label]) if label else "[]"
    with path.open("a") as stream:
        stream.write(f"has_device={'true' if lease is not None else 'false'}\n")
        stream.write(f"runs_on={runs_on}\n")


def acquire_plan_leases(
    plan: Mapping[str, Any],
    devices: Sequence[Device],
    store: GitHubRefLeaseStore,
    *,
    attempt_id: str,
    head_sha: str,
    target_sha: str,
    run_url: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    planned_work = work_items(plan)
    ordered = sorted(
        enumerate(planned_work),
        key=lambda item: (-required_memory_gib(item[1]), item[0]),
    )
    leases_by_label: dict[str, DeviceLease] = {}
    queue_depth = {device.label: 0 for device in devices}
    records: list[dict[str, Any]] = []
    for sequence, (_, work_item) in enumerate(ordered):
        dispatch = create_dispatch(work_item, devices)
        dispatch["candidates"] = sorted(
            dispatch["candidates"],
            key=lambda candidate: (
                queue_depth.get(str(candidate.get("label")), 0),
                int(candidate.get("memory_gib", 0)),
            ),
        )
        dispatch, lease = _acquire_or_reuse_dispatch(
            dispatch,
            store,
            leases_by_label,
            attempt_id=attempt_id,
            head_sha=head_sha,
            target_sha=target_sha,
            run_url=run_url,
            ttl_seconds=ttl_seconds,
            now=now,
        )
        subject = (
            work_item.get("model")
            or work_item.get("profile")
            or work_item.get("work_type")
            or "work"
        )
        key = f"{sequence:03d}-{_safe_key(str(subject))}"
        records.append(
            {
                "key": key,
                "execute": lease is not None,
                "work": dict(dispatch["job"]),
                "dispatch": dispatch,
                "lease": (
                    asdict(replace(lease, release_on_job_end=False))
                    if lease is not None
                    else None
                ),
            }
        )
        if lease is not None:
            leases_by_label[lease.label] = lease
            queue_depth[lease.label] += 1
    return {
        "schema_version": 1,
        "kind": "device_dispatch_batch",
        "attempt_id": attempt_id,
        "head_sha": head_sha,
        "items": records,
    }


def write_batch(directory: Path, batch: Mapping[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    matrix: list[dict[str, Any]] = []
    for item in batch.get("items", []):
        if not isinstance(item, Mapping):
            raise DeviceLeaseError("dispatch batch item must be an object")
        key = str(item.get("key", ""))
        item_directory = directory / key
        item_directory.mkdir(parents=True, exist_ok=True)
        for name in ("work", "dispatch", "lease"):
            value = item.get(name)
            (item_directory / f"{name}.json").write_text(
                json.dumps(value or {}, indent=2, sort_keys=True) + "\n"
            )
        lease = item.get("lease")
        if (
            item.get("execute", True)
            and isinstance(lease, Mapping)
            and lease.get("label")
        ):
            matrix.append(
                {
                    "key": key,
                    "runs_on": ["self-hosted", str(lease["label"])],
                }
            )
    (directory / "batch.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n"
    )
    return {"include": matrix}


def write_batch_github_output(path: Path, matrix: Mapping[str, Any]) -> None:
    include = matrix.get("include", [])
    with path.open("a") as stream:
        stream.write(f"has_work={'true' if include else 'false'}\n")
        stream.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")


def release_batch(
    batch: Mapping[str, Any], store: GitHubRefLeaseStore
) -> dict[str, bool]:
    released: dict[str, bool] = {}
    for item in batch.get("items", []):
        if not isinstance(item, Mapping):
            continue
        lease = item.get("lease")
        if not isinstance(lease, Mapping) or not lease:
            continue
        value = DeviceLease(**lease)
        released[value.device] = store.release(value.device, value.attempt_id)
    return released


def release_job(lease: DeviceLease, store: GitHubRefLeaseStore) -> bool:
    if not lease.release_on_job_end:
        return True
    return store.release(lease.device, lease.attempt_id)


def retry_batch(
    batch: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    devices: Sequence[Device],
    store: GitHubRefLeaseStore,
    *,
    attempt_id: str,
    head_sha: str,
    target_sha: str,
    run_url: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    leases_by_label = {
        str(lease["label"]): DeviceLease(**lease)
        for raw_item in batch.get("items", [])
        if isinstance(raw_item, Mapping)
        and isinstance((lease := raw_item.get("lease")), Mapping)
        and lease
    }
    items: list[dict[str, Any]] = []
    for raw_item in batch.get("items", []):
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        key = str(item.get("key", ""))
        result = results.get(key)
        lease = item.get("lease")
        if result is not None and result.get("decision") == "accepted":
            item["execute"] = False
            items.append(item)
            continue
        tried: list[dict[str, Any]] = []
        minimum_memory = 0
        if isinstance(lease, Mapping) and lease:
            old_lease = DeviceLease(**lease)
            minimum_memory = next(
                (
                    device.memory_gib
                    for device in devices
                    if device.name == old_lease.device
                ),
                0,
            )
            tried.append(
                {
                    "device": old_lease.device,
                    "label": old_lease.label,
                    "memory_gib": minimum_memory,
                    "decision": (
                        str(result.get("decision", "declined"))
                        if result is not None
                        else "declined"
                    ),
                    "reason": (
                        str(result.get("reason", "runner_disappeared"))
                        if result is not None
                        else "runner_disappeared"
                    ),
                    "details": (
                        dict(result.get("observed", {}))
                        if result is not None
                        and isinstance(result.get("observed"), Mapping)
                        else {}
                    ),
                }
            )
        candidates = [
            device
            for device in devices
            if device.memory_gib > minimum_memory
            and all(device.name != attempt["device"] for attempt in tried)
        ]
        work = item.get("work")
        if not isinstance(work, Mapping):
            raise DeviceLeaseError("dispatch batch item has no work object")
        dispatch = create_dispatch(work, candidates)
        dispatch["attempts"] = tried
        dispatch, new_lease = _acquire_or_reuse_dispatch(
            dispatch,
            store,
            leases_by_label,
            attempt_id=attempt_id,
            head_sha=head_sha,
            target_sha=target_sha,
            run_url=run_url,
            ttl_seconds=ttl_seconds,
            now=now,
        )
        item.update(
            {
                "execute": new_lease is not None,
                "dispatch": dispatch,
                "lease": (
                    asdict(replace(new_lease, release_on_job_end=False))
                    if new_lease is not None
                    else None
                ),
            }
        )
        if new_lease is not None:
            leases_by_label[new_lease.label] = new_lease
        items.append(item)
    return {
        "schema_version": 1,
        "kind": "device_dispatch_batch",
        "attempt_id": attempt_id,
        "head_sha": head_sha,
        "items": items,
    }


def _acquire_or_reuse_dispatch(
    dispatch: Mapping[str, Any],
    store: GitHubRefLeaseStore,
    leases_by_label: Mapping[str, DeviceLease],
    *,
    attempt_id: str,
    head_sha: str,
    target_sha: str,
    run_url: str,
    ttl_seconds: int,
    now: datetime | None,
) -> tuple[dict[str, Any], DeviceLease | None]:
    candidates = dispatch.get("candidates")
    if not isinstance(candidates, list):
        raise DeviceLeaseError("dispatch has no candidate list")
    updated = dict(dispatch)
    unavailable = list(dispatch.get("unavailable", []))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise DeviceLeaseError("dispatch candidate must be an object")
        label = str(candidate.get("label", ""))
        lease = leases_by_label.get(label)
        if lease is None:
            lease = store.acquire(
                attempt_id=attempt_id,
                device=str(candidate.get("name", "")),
                label=label,
                head_sha=head_sha,
                target_sha=target_sha,
                run_url=run_url,
                ttl_seconds=ttl_seconds,
                now=now,
            )
        if lease is not None:
            updated["next_device"] = dict(candidate)
            updated["lease"] = asdict(lease)
            updated["outcome"] = "dispatching"
            updated["unavailable"] = unavailable
            return updated, lease
        owner = store.get(str(candidate.get("name", "")))
        record = {
            "device": candidate.get("name"),
            "memory_gib": candidate.get("memory_gib"),
            "reason": "leased",
        }
        if owner is not None:
            record.update(
                {"attempt_id": owner.attempt_id, "expires_at": owner.expires_at}
            )
        unavailable.append(record)
    updated["next_device"] = None
    updated["lease"] = None
    updated["outcome"] = "no_eligible_runner"
    updated["unavailable"] = unavailable
    return updated, None


def _safe_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized[:80] or "model"


def _new_lease(
    *,
    attempt_id: str,
    device: str,
    label: str,
    head_sha: str,
    target_sha: str,
    run_url: str,
    ttl_seconds: int,
    now: datetime,
) -> DeviceLease:
    if ttl_seconds <= 0:
        raise DeviceLeaseError("lease TTL must be positive")
    _ref_name(device)
    if not attempt_id or not label or not head_sha or not target_sha:
        raise DeviceLeaseError("lease identity fields must not be empty")
    return DeviceLease(
        schema_version=1,
        attempt_id=attempt_id,
        device=device,
        label=label,
        head_sha=head_sha,
        target_sha=target_sha,
        run_url=run_url,
        acquired_at=_format_time(now),
        heartbeat_at=_format_time(now),
        expires_at=_format_time(now + timedelta(seconds=ttl_seconds)),
        generation=uuid.uuid4().hex,
    )


def _ref_name(device: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", device):
        raise DeviceLeaseError("device name is not safe for a Git ref")
    return f"{REF_PREFIX}/{device}"


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise DeviceLeaseError("lease time must include a timezone")
    return value.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeviceLeaseError("lease contains an invalid timestamp") from error
    return _utc(parsed)


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise DeviceLeaseError(f"{path} must contain an object")
    return value


def _heartbeat_loop(
    store: GitHubRefLeaseStore,
    lease: DeviceLease,
    ttl_seconds: int,
    interval_seconds: int,
    parent_pid: int,
    stop_file: Path,
) -> int:
    while not stop_file.exists() and _process_exists(parent_pid):
        store.heartbeat(lease.device, lease.attempt_id, ttl_seconds)
        for _ in range(interval_seconds):
            if stop_file.exists() or not _process_exists(parent_pid):
                return 0
            time.sleep(1)
    return 0


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--dispatch", type=Path, required=True)
    acquire_parser.add_argument("--repository", required=True)
    acquire_parser.add_argument("--attempt-id", required=True)
    acquire_parser.add_argument("--head-sha", required=True)
    acquire_parser.add_argument("--target-sha", required=True)
    acquire_parser.add_argument("--run-url", required=True)
    acquire_parser.add_argument("--ttl-seconds", type=int, default=300)
    acquire_parser.add_argument("--lease", type=Path, required=True)
    acquire_parser.add_argument("--github-output", type=Path, required=True)

    batch_parser = subparsers.add_parser("acquire-batch")
    batch_parser.add_argument("--plan", type=Path, required=True)
    batch_parser.add_argument("--devices", type=Path, required=True)
    batch_parser.add_argument("--busy", type=Path, required=True)
    batch_parser.add_argument("--live-runners", type=Path)
    batch_parser.add_argument("--repository", required=True)
    batch_parser.add_argument("--attempt-id", required=True)
    batch_parser.add_argument("--head-sha", required=True)
    batch_parser.add_argument("--target-sha", required=True)
    batch_parser.add_argument("--run-url", required=True)
    batch_parser.add_argument("--ttl-seconds", type=int, default=300)
    batch_parser.add_argument("--directory", type=Path, required=True)
    batch_parser.add_argument("--github-output", type=Path, required=True)

    retry_parser = subparsers.add_parser("retry-batch")
    retry_parser.add_argument("--batch", type=Path, required=True)
    retry_parser.add_argument("--results-directory", type=Path, required=True)
    retry_parser.add_argument("--devices", type=Path, required=True)
    retry_parser.add_argument("--busy", type=Path, required=True)
    retry_parser.add_argument("--live-runners", type=Path, required=True)
    retry_parser.add_argument("--repository", required=True)
    retry_parser.add_argument("--attempt-id", required=True)
    retry_parser.add_argument("--head-sha", required=True)
    retry_parser.add_argument("--target-sha", required=True)
    retry_parser.add_argument("--run-url", required=True)
    retry_parser.add_argument("--ttl-seconds", type=int, default=300)
    retry_parser.add_argument("--directory", type=Path, required=True)
    retry_parser.add_argument("--github-output", type=Path, required=True)

    heartbeat_parser = subparsers.add_parser("heartbeat-loop")
    heartbeat_parser.add_argument("--lease", type=Path, required=True)
    heartbeat_parser.add_argument("--repository", required=True)
    heartbeat_parser.add_argument("--ttl-seconds", type=int, default=300)
    heartbeat_parser.add_argument("--interval-seconds", type=int, default=30)
    heartbeat_parser.add_argument("--parent-pid", type=int, required=True)
    heartbeat_parser.add_argument("--stop-file", type=Path, required=True)

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--lease", type=Path, required=True)
    release_parser.add_argument("--repository", required=True)

    release_batch_parser = subparsers.add_parser("release-batch")
    release_batch_parser.add_argument("--batch", type=Path, required=True)
    release_batch_parser.add_argument("--repository", required=True)

    args = parser.parse_args(argv)
    store = GitHubRefLeaseStore(args.repository)
    if args.command == "acquire":
        dispatch, lease = acquire_dispatch_lease(
            _load_object(args.dispatch),
            store,
            attempt_id=args.attempt_id,
            head_sha=args.head_sha,
            target_sha=args.target_sha,
            run_url=args.run_url,
            ttl_seconds=args.ttl_seconds,
        )
        args.dispatch.write_text(json.dumps(dispatch, indent=2, sort_keys=True) + "\n")
        args.lease.write_text(
            json.dumps(
                asdict(lease) if lease is not None else {}, indent=2, sort_keys=True
            )
            + "\n"
        )
        write_github_output(args.github_output, lease)
        return 0
    if args.command == "acquire-batch":
        device_config = json.loads(args.devices.read_text())
        busy = json.loads(args.busy.read_text())
        if not isinstance(device_config, list):
            raise DeviceLeaseError("devices must contain a list")
        if not isinstance(busy, list) or any(
            not isinstance(name, str) for name in busy
        ):
            raise DeviceLeaseError("busy runners must contain a string list")
        batch = acquire_plan_leases(
            _load_object(args.plan),
            configured_devices(
                device_config,
                set(busy),
                _load_object(args.live_runners) if args.live_runners else None,
            ),
            store,
            attempt_id=args.attempt_id,
            head_sha=args.head_sha,
            target_sha=args.target_sha,
            run_url=args.run_url,
            ttl_seconds=args.ttl_seconds,
        )
        matrix = write_batch(args.directory, batch)
        write_batch_github_output(args.github_output, matrix)
        return 0
    if args.command == "retry-batch":
        device_config = json.loads(args.devices.read_text())
        busy = json.loads(args.busy.read_text())
        if not isinstance(device_config, list) or not isinstance(busy, list):
            raise DeviceLeaseError("retry inventory is invalid")
        results = {
            path.stem.removeprefix("result-"): _load_object(path)
            for path in args.results_directory.glob("result-*.json")
        }
        batch = retry_batch(
            _load_object(args.batch),
            results,
            configured_devices(
                device_config,
                set(str(name) for name in busy),
                _load_object(args.live_runners),
            ),
            store,
            attempt_id=args.attempt_id,
            head_sha=args.head_sha,
            target_sha=args.target_sha,
            run_url=args.run_url,
            ttl_seconds=args.ttl_seconds,
        )
        matrix = write_batch(args.directory, batch)
        write_batch_github_output(args.github_output, matrix)
        return 0
    if args.command == "release-batch":
        if args.batch.is_file():
            release_batch(_load_object(args.batch), store)
        return 0
    if not args.lease.is_file():
        return 0
    payload = _load_object(args.lease)
    if not payload:
        return 0
    lease = DeviceLease(**payload)
    if args.command == "heartbeat-loop":
        return _heartbeat_loop(
            store,
            lease,
            args.ttl_seconds,
            args.interval_seconds,
            args.parent_pid,
            args.stop_file,
        )
    release_job(lease, store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
