from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.device_lease import (
    UPDATE_REFS_MUTATION,
    ZERO_OID,
    DeviceLeaseError,
    GhClient,
    GitHubApiError,
    GitHubClient,
    _format_time,
    _parse_time,
    _utc,
)

REF_PREFIX = "refs/tags/ci-attempt-lease"


@dataclass(frozen=True)
class AttemptLease:
    schema_version: int
    attempt_id: str
    pr_number: int
    head_sha: str
    target_sha: str
    run_url: str
    acquired_at: str
    expires_at: str
    generation: str
    token: str = ""

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token")
        return value

    def expired(self, now: datetime) -> bool:
        return _parse_time(self.expires_at) <= now


class AttemptLeaseStore:
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
        pr_number: int,
        head_sha: str,
        target_sha: str,
        run_url: str,
        ttl_seconds: int,
        now: datetime | None = None,
    ) -> tuple[AttemptLease, bool]:
        instant = _utc(now)
        current = self.get(pr_number, head_sha)
        if current is not None and not current.expired(instant):
            return current, False
        if ttl_seconds <= 0 or pr_number <= 0:
            raise DeviceLeaseError("attempt lease configuration is invalid")
        candidate = AttemptLease(
            schema_version=1,
            attempt_id=attempt_id,
            pr_number=pr_number,
            head_sha=head_sha,
            target_sha=target_sha,
            run_url=run_url,
            acquired_at=_format_time(instant),
            expires_at=_format_time(instant + timedelta(seconds=ttl_seconds)),
            generation=uuid.uuid4().hex,
        )
        tag_oid = self._create_tag(candidate)
        before_oid = current.token if current is not None else ZERO_OID
        try:
            self._update_ref(candidate, before_oid, tag_oid)
        except GitHubApiError:
            refreshed = self.get(pr_number, head_sha)
            if refreshed is not None and refreshed.token != before_oid:
                return refreshed, False
            raise
        return AttemptLease(**candidate.payload(), token=tag_oid), True

    def get(self, pr_number: int, head_sha: str) -> AttemptLease | None:
        ref = _ref_name(pr_number, head_sha)
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
            raise DeviceLeaseError("attempt lease ref has no target object")
        token = str(target["sha"])
        tag = self.client.rest(f"repos/{self.repository}/git/tags/{token}")
        payload = json.loads(str(tag.get("message", "")))
        if not isinstance(payload, Mapping):
            raise DeviceLeaseError("attempt lease payload must be an object")
        lease = AttemptLease(**payload, token=token)
        if (
            lease.schema_version != 1
            or lease.pr_number != pr_number
            or lease.head_sha != head_sha
        ):
            raise DeviceLeaseError("attempt lease payload does not match its ref")
        return lease

    def release(self, lease: AttemptLease) -> bool:
        current = self.get(lease.pr_number, lease.head_sha)
        if current is None:
            return True
        if current.attempt_id != lease.attempt_id:
            return False
        try:
            self._update_ref(current, current.token, ZERO_OID)
        except GitHubApiError:
            return False
        return True

    def _create_tag(self, lease: AttemptLease) -> str:
        value = self.client.rest(
            f"repos/{self.repository}/git/tags",
            method="POST",
            body={
                "tag": f"ci-attempt-{lease.pr_number}-{lease.attempt_id}-{lease.generation[:12]}",
                "message": json.dumps(
                    lease.payload(), sort_keys=True, separators=(",", ":")
                ),
                "object": lease.target_sha,
                "type": "commit",
            },
        )
        oid = value.get("sha")
        if not isinstance(oid, str) or not oid:
            raise DeviceLeaseError("GitHub did not return an attempt tag object ID")
        return oid

    def _update_ref(self, lease: AttemptLease, before_oid: str, after_oid: str) -> None:
        self.client.graphql(
            UPDATE_REFS_MUTATION,
            {
                "input": {
                    "repositoryId": self._repository_id(),
                    "refUpdates": [
                        {
                            "name": _ref_name(lease.pr_number, lease.head_sha),
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


def _ref_name(pr_number: int, head_sha: str) -> str:
    if pr_number <= 0 or not re.fullmatch(r"[0-9a-fA-F]{7,64}", head_sha):
        raise DeviceLeaseError("attempt lease identity is invalid")
    return f"{REF_PREFIX}/{pr_number}-{head_sha.lower()}"


def _load(path: Path) -> AttemptLease:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise DeviceLeaseError("attempt lease file must contain an object")
    return AttemptLease(**value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--repository", required=True)
    acquire.add_argument("--attempt-id", required=True)
    acquire.add_argument("--pr", type=int, required=True)
    acquire.add_argument("--head-sha", required=True)
    acquire.add_argument("--target-sha", required=True)
    acquire.add_argument("--run-url", required=True)
    acquire.add_argument("--ttl-seconds", type=int, default=7200)
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--github-output", type=Path, required=True)
    release = subparsers.add_parser("release")
    release.add_argument("--repository", required=True)
    release.add_argument("--lease", type=Path, required=True)
    args = parser.parse_args(argv)
    store = AttemptLeaseStore(args.repository)
    if args.command == "acquire":
        lease, acquired = store.acquire(
            attempt_id=args.attempt_id,
            pr_number=args.pr,
            head_sha=args.head_sha,
            target_sha=args.target_sha,
            run_url=args.run_url,
            ttl_seconds=args.ttl_seconds,
        )
        args.output.write_text(
            json.dumps(asdict(lease), indent=2, sort_keys=True) + "\n"
        )
        with args.github_output.open("a") as stream:
            stream.write(f"acquired={'true' if acquired else 'false'}\n")
            stream.write(f"owner_attempt_id={lease.attempt_id}\n")
            stream.write(f"owner_run_url={lease.run_url}\n")
        return 0
    if args.lease.is_file():
        store.release(_load(args.lease))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
