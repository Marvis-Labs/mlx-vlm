from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

GIB = 1 << 30
FAMILY = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
MODEL_PREFIX = ("mlx_vlm", "models")
SERVER_PREFIX = ("mlx_vlm", "server")
SERVER_TEST = "mlx_vlm/tests/test_server.py"
SERVER_PROFILES = {
    "anthropic.py": "anthropic",
    "audio.py": "audio",
    "embeddings.py": "embeddings",
    "generation.py": "generation",
    "model_discovery.py": "model_discovery",
    "openai.py": "openai",
    "realtime.py": "realtime",
    "request_normalization.py": "openai",
    "reranking.py": "reranking",
    "responses_state.py": "openai",
    "runtime.py": "runtime",
    "runtime_config.py": "runtime",
    "schemas.py": "openai",
}
CATALOG = Path(__file__).resolve().parents[1] / "mlx_vlm" / "tests" / "model_cases.json"


class ModelPlanError(ValueError):
    pass


def _paths(changed_files: Iterable[str]) -> tuple[str, ...]:
    paths = []
    for changed in changed_files:
        if not isinstance(changed, str) or "\0" in changed:
            raise ModelPlanError("changed file is invalid")
        paths.append(changed)
    return tuple(paths)


def _families(changed_files: Iterable[str]) -> list[str]:
    families = set()
    for changed in changed_files:
        parts = PurePosixPath(changed).parts
        if len(parts) >= 4 and parts[:2] == MODEL_PREFIX:
            family = parts[2]
            if FAMILY.fullmatch(family) is None:
                raise ModelPlanError("model family is invalid")
            families.add(family)
    return sorted(families)


def _server_profiles(changed_files: Iterable[str]) -> list[str]:
    profiles = set()
    for changed in changed_files:
        if changed == SERVER_TEST:
            return ["all"]
        parts = PurePosixPath(changed).parts
        if len(parts) == 3 and parts[:2] == SERVER_PREFIX:
            profiles.add(SERVER_PROFILES.get(parts[2], "core"))
    return sorted(profiles)


def _synthetic_tests(catalog: dict[str, Any]) -> dict[str, list[str]]:
    tests: dict[str, list[str]] = {}
    for case in catalog.get("cases", []):
        family, case_id = case.get("module"), case.get("id")
        if not isinstance(family, str) or not isinstance(case_id, str):
            raise ModelPlanError("model case is invalid")
        tests.setdefault(family, []).append(
            f"mlx_vlm/tests/test_models.py::test_model_contract[{case_id}]"
        )
    dense = catalog.get("dense", {})
    if not isinstance(dense, dict):
        raise ModelPlanError("dense model cases are invalid")
    for family in dense:
        tests.setdefault(family, []).append(
            f"mlx_vlm/tests/test_models.py::test_dense_model[{family}]"
        )
    return tests


def _resources(checkpoint: dict[str, Any] | None) -> dict[str, int]:
    resident = checkpoint["tensor_bytes"] if checkpoint else 256 << 20
    return {
        "resident_bytes": resident,
        "fixed_bytes": (2 if checkpoint else 1) * GIB,
        "bytes_per_unit": (2 << 20) if checkpoint else (256 << 10),
        "units": 512,
        "batch_size": 1,
        "workspace_bytes": 4 * GIB,
    }


def plan_ci(changed_files: Iterable[str], catalog: dict[str, Any]) -> dict[str, Any]:
    """Return repository CI work for a set of changed paths."""
    if not isinstance(catalog, dict) or catalog.get("version") != 2:
        raise ModelPlanError("unsupported model catalog")
    changed_files = _paths(changed_files)
    tests = _synthetic_tests(catalog)
    ci = catalog.get("ci", {})
    checkpoints = ci.get("checkpoints", {})
    profiles = ci.get("checkpoint_profiles", {})
    default_profile = ci.get("default_checkpoint_profile")
    if not isinstance(checkpoints, dict):
        raise ModelPlanError("checkpoint catalog is invalid")
    if (
        not isinstance(profiles, dict)
        or not isinstance(default_profile, str)
        or default_profile not in profiles
    ):
        raise ModelPlanError("checkpoint profiles are invalid")
    jobs, blocked = [], []
    for family in _families(changed_files):
        selectors = tests.get(family)
        if not selectors:
            blocked.append(
                {
                    "component": "model_path",
                    "subject": family,
                    "reason": "model_case_missing",
                }
            )
            continue
        checkpoint = checkpoints.get(family)
        if checkpoint is not None:
            if (
                not isinstance(checkpoint, dict)
                or not {"repository", "revision", "tensor_bytes"}.issubset(checkpoint)
                or set(checkpoint)
                - {"repository", "revision", "tensor_bytes", "profile"}
                or not isinstance(checkpoint["tensor_bytes"], int)
                or checkpoint["tensor_bytes"] <= 0
            ):
                raise ModelPlanError(f"checkpoint for {family} is invalid")
            profile_name = checkpoint.get("profile", default_profile)
            profile = profiles.get(profile_name)
            if not isinstance(profile, dict):
                raise ModelPlanError(f"checkpoint profile for {family} is invalid")
        phases = ["synthetic"]
        work: dict[str, Any] = {"synthetic": {"selectors": selectors}}
        artifact = None
        if checkpoint:
            phases.append("checkpoint")
            work["checkpoint"] = {"profile": profile_name, **profile}
            artifact = {
                "kind": "huggingface",
                **{
                    key: checkpoint[key]
                    for key in ("repository", "revision", "tensor_bytes")
                },
            }
        jobs.append(
            {
                "id": f"model-path-{family}",
                "component": "model_path",
                "subject": family,
                "phases": phases,
                "work": work,
                "resources": _resources(checkpoint),
                "artifact": artifact,
            }
        )
    profiles = _server_profiles(changed_files)
    if profiles:
        jobs.append(
            {
                "id": "server-change",
                "component": "server_change",
                "subject": "server",
                "phases": ["server_contract"],
                "work": {
                    "server_contract": {
                        "profiles": profiles,
                        "selectors": [SERVER_TEST],
                    }
                },
                "resources": {
                    "resident_bytes": 256 << 20,
                    "fixed_bytes": 1 * GIB,
                    "bytes_per_unit": 0,
                    "units": 0,
                    "batch_size": 1,
                    "workspace_bytes": 2 * GIB,
                },
                "artifact": None,
            }
        )
    return {"schema_version": 1, "jobs": jobs, "blocked": blocked}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--catalog", default=CATALOG, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    attempt = json.loads(arguments.attempt.read_text(encoding="utf-8"))
    catalog = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    changed_files = attempt.get("changed_files")
    if not isinstance(changed_files, list):
        raise ModelPlanError("attempt has no changed files")
    _write_json(arguments.output, plan_ci(changed_files, catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
