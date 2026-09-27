from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
JOB_FIELDS = {
    "schema_version",
    "engine",
    "repository",
    "pull_request",
    "base_sha",
    "head_sha",
    "head_repository",
    "contract_sha",
    "id",
    "component",
    "subject",
    "phases",
    "work",
    "resources",
    "artifact",
    "estimated_peak_bytes",
    "required_memory_gib",
    "required_disk_gib",
    "manifest_digest",
}
PHASES = {
    "model_path": frozenset({"synthetic", "checkpoint"}),
    "server_change": frozenset({"server_contract"}),
}


class ExecutionSecurityError(ValueError):
    pass


def validate_job(job: Mapping[str, Any]) -> None:
    """Validate the complete sealed work manifest accepted by the runner."""
    if not isinstance(job, Mapping) or set(job) != JOB_FIELDS:
        raise ExecutionSecurityError("work manifest fields are invalid")
    if job["schema_version"] != 2 or type(job["schema_version"]) is not int:
        raise ExecutionSecurityError("work manifest version is invalid")
    for field in ("engine", "id", "component", "subject"):
        if not isinstance(job[field], str) or NAME.fullmatch(job[field]) is None:
            raise ExecutionSecurityError(f"work manifest {field} is invalid")
    for field in ("repository", "head_repository"):
        if not isinstance(job[field], str) or REPOSITORY.fullmatch(job[field]) is None:
            raise ExecutionSecurityError(f"work manifest {field} is invalid")
    for field in ("base_sha", "head_sha", "contract_sha"):
        if not isinstance(job[field], str) or SHA.fullmatch(job[field]) is None:
            raise ExecutionSecurityError(f"work manifest {field} is invalid")
    if job["contract_sha"] != job["base_sha"]:
        raise ExecutionSecurityError("work manifest contract is not current main")
    phases = job["phases"]
    allowed = PHASES.get(job["component"])
    if (
        allowed is None
        or not isinstance(phases, list)
        or not phases
        or len(phases) != len(set(phases))
        or any(phase not in allowed for phase in phases)
    ):
        raise ExecutionSecurityError("work manifest phases are invalid")
    if not isinstance(job["work"], Mapping) or set(job["work"]) != set(phases):
        raise ExecutionSecurityError("work manifest phase configuration is invalid")
    for field in ("pull_request", "required_memory_gib", "required_disk_gib"):
        if type(job[field]) is not int or job[field] <= 0:
            raise ExecutionSecurityError(f"work manifest {field} is invalid")
    digest = job["manifest_digest"]
    if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
        raise ExecutionSecurityError("work manifest digest is invalid")
    unsigned = {key: value for key, value in job.items() if key != "manifest_digest"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ExecutionSecurityError("work manifest digest does not match")


def verify_execution(
    job: Mapping[str, Any], control: Path, base: Path, head: Path
) -> None:
    """Bind execution to clean checkouts of the three sealed revisions."""
    validate_job(job)
    for path, expected, label in (
        (control, job["contract_sha"], "control"),
        (base, job["base_sha"], "base"),
        (head, job["head_sha"], "head"),
    ):
        if path.is_symlink() or not path.resolve(strict=True).is_dir():
            raise ExecutionSecurityError(f"{label} checkout is invalid")
        if _git(path, "rev-parse", "--verify", "HEAD^{commit}") != expected:
            raise ExecutionSecurityError(f"{label} checkout revision does not match")
        if _git(path, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ExecutionSecurityError(f"{label} checkout is not clean")


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
