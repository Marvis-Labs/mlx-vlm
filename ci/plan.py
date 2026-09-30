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
SCENARIO = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
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


def _model_tests(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tests: dict[str, dict[str, Any]] = {}
    for case in catalog.get("cases", []):
        family, case_id = case.get("module"), case.get("id")
        if not isinstance(family, str) or not isinstance(case_id, str):
            raise ModelPlanError("model case is invalid")
        model = tests.setdefault(
            family, {"selectors": [], "checks": set(), "batch_size": 1}
        )
        model["selectors"].append(
            f"mlx_vlm/tests/test_models.py::test_model_contract[{case_id}]"
        )
        model["checks"].update(case.get("checks", []))
        batch_sizes = case.get("multimodal", {}).get("batch_sizes", [])
        if batch_sizes:
            model["batch_size"] = max(model["batch_size"], *batch_sizes)
    dense = catalog.get("dense", {})
    if not isinstance(dense, dict):
        raise ModelPlanError("dense model cases are invalid")
    for family in dense:
        model = tests.setdefault(
            family, {"selectors": [], "checks": set(), "batch_size": 1}
        )
        model["selectors"].append(
            f"mlx_vlm/tests/test_models.py::test_dense_model[{family}]"
        )
    return tests


def _resources(
    checkpoint: dict[str, Any] | None,
    scenario: dict[str, Any] | None = None,
    batch_size: int = 1,
) -> dict[str, int]:
    resident = checkpoint["tensor_bytes"] if checkpoint else 256 << 20
    scenario = scenario or {}
    return {
        "resident_bytes": resident,
        "fixed_bytes": (2 if checkpoint else 1) * GIB,
        "bytes_per_unit": (2 << 20) if checkpoint else (256 << 10),
        "units": int(scenario.get("prompt_tokens", 512)),
        "batch_size": max(batch_size, int(scenario.get("batch_size", 1))),
        "workspace_bytes": 4 * GIB,
    }


def _assets(ci: dict[str, Any]) -> dict[str, str]:
    assets = ci.get("assets")
    if not isinstance(assets, dict):
        raise ModelPlanError("asset catalog is invalid")
    for asset_id, relative in assets.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or path is None
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
        ):
            raise ModelPlanError("asset catalog is invalid")
    return assets


def _scenario(scenario_id: str, value: Any, assets: dict[str, str]) -> dict[str, Any]:
    if SCENARIO.fullmatch(scenario_id) is None or not isinstance(value, dict):
        raise ModelPlanError(f"scenario {scenario_id} is invalid")
    executor = value.get("executor")
    batch_size = value.get("batch_size", 1)
    prompt_tokens = value.get("prompt_tokens")
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= 8
        or type(prompt_tokens) is not int
        or not 1 <= prompt_tokens <= 8192
    ):
        raise ModelPlanError(f"scenario {scenario_id} is invalid")
    if executor == "generation":
        expected = {
            "executor",
            "inputs",
            "prompt",
            "prompt_tokens",
            "max_tokens",
            "batch_size",
            "contains",
        }
        inputs = value.get("inputs")
        if (
            set(value) - expected
            or not isinstance(value.get("prompt"), str)
            or not value["prompt"]
            or type(value.get("max_tokens")) is not int
            or not 1 <= value["max_tokens"] <= 128
            or not isinstance(value.get("contains", ""), str)
            or not isinstance(inputs, dict)
            or set(inputs) - {"image", "audio"}
            or any(
                not isinstance(asset, str) or asset not in assets
                for asset in inputs.values()
            )
        ):
            raise ModelPlanError(f"scenario {scenario_id} is invalid")
        value = value | {
            "inputs": {kind: assets[asset] for kind, asset in inputs.items()}
        }
    elif executor == "embedding":
        expected = {
            "executor",
            "embedding_kind",
            "texts",
            "prompt_tokens",
            "batch_size",
        }
        if (
            set(value) - expected
            or value.get("embedding_kind") not in {"sentence", "token"}
            or not isinstance(value.get("texts"), list)
            or len(value["texts"]) < 3
            or any(not isinstance(text, str) or not text for text in value["texts"])
        ):
            raise ModelPlanError(f"scenario {scenario_id} is invalid")
    elif executor == "rerank":
        expected = {
            "executor",
            "query",
            "instruction",
            "documents",
            "prompt_tokens",
            "batch_size",
        }
        documents = value.get("documents")
        if (
            set(value) - expected
            or not isinstance(value.get("query"), str)
            or not value["query"]
            or not isinstance(value.get("instruction"), str)
            or not value["instruction"]
            or not isinstance(documents, list)
            or len(documents) < 2
            or any(
                not isinstance(document, str) or not document for document in documents
            )
        ):
            raise ModelPlanError(f"scenario {scenario_id} is invalid")
    else:
        raise ModelPlanError(f"scenario {scenario_id} is invalid")
    return value


def _checkpoint(family: str, value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"repository", "revision", "tensor_bytes"}
        or not isinstance(value["repository"], str)
        or not value["repository"]
        or REVISION.fullmatch(str(value["revision"])) is None
        or type(value["tensor_bytes"]) is not int
        or value["tensor_bytes"] <= 0
    ):
        raise ModelPlanError(f"checkpoint for {family} is invalid")
    return value


def plan_ci(changed_files: Iterable[str], catalog: dict[str, Any]) -> dict[str, Any]:
    """Return repository CI work for a set of changed paths."""
    if not isinstance(catalog, dict) or catalog.get("version") != 2:
        raise ModelPlanError("unsupported model catalog")
    changed_files = _paths(changed_files)
    tests = _model_tests(catalog)
    ci = catalog.get("ci", {})
    assets = _assets(ci)
    raw_scenarios = ci.get("scenarios")
    model_paths = ci.get("model_paths")
    if not isinstance(raw_scenarios, dict) or not isinstance(model_paths, dict):
        raise ModelPlanError("model path catalog is invalid")
    scenarios = {
        scenario_id: _scenario(scenario_id, value, assets)
        for scenario_id, value in raw_scenarios.items()
    }
    jobs, blocked = [], []
    for family in _families(changed_files):
        model_test = tests.get(family)
        if not model_test:
            blocked.append(
                {
                    "component": "model_path",
                    "subject": family,
                    "reason": "model_case_missing",
                }
            )
            continue
        selectors = model_test["selectors"]
        model_checks = sorted(model_test["checks"])
        test_batch_size = model_test["batch_size"]
        jobs.append(
            {
                "id": f"model-path-{family}-synthetic",
                "component": "model_path",
                "subject": family,
                "phases": ["synthetic"],
                "work": {"synthetic": {"selectors": selectors}},
                "resources": _resources(None, None, test_batch_size),
                "artifact": None,
            }
        )
        registrations = model_paths.get(family, [])
        if isinstance(registrations, dict):
            registrations = [registrations]
        if not isinstance(registrations, list):
            raise ModelPlanError(f"model path for {family} is invalid")
        seen = set()
        for registration in registrations:
            scenario_field = (
                "scenarios"
                if isinstance(registration, dict) and "scenarios" in registration
                else "scenario"
            )
            fields = {scenario_field, "repository", "revision", "tensor_bytes"}
            if not isinstance(registration, dict) or set(registration) != fields:
                raise ModelPlanError(f"model path for {family} is invalid")
            scenario_ids = registration[scenario_field]
            if scenario_field == "scenario":
                scenario_ids = [scenario_ids]
            if not isinstance(scenario_ids, list) or not scenario_ids:
                raise ModelPlanError(f"model path for {family} is invalid")
            checkpoint = _checkpoint(
                family,
                {
                    key: registration[key]
                    for key in ("repository", "revision", "tensor_bytes")
                },
            )
            artifact = {
                "kind": "huggingface",
                **{
                    key: checkpoint[key]
                    for key in ("repository", "revision", "tensor_bytes")
                },
            }
            for scenario_id in scenario_ids:
                if (
                    not isinstance(scenario_id, str)
                    or scenario_id in seen
                    or scenario_id not in scenarios
                ):
                    raise ModelPlanError(f"model path for {family} is invalid")
                seen.add(scenario_id)
                scenario = scenarios[scenario_id]
                work = {
                    "scenario": scenario_id,
                    "model_checks": model_checks,
                    **scenario,
                    "batch_size": max(
                        test_batch_size, int(scenario.get("batch_size", 1))
                    ),
                }
                jobs.append(
                    {
                        "id": f"model-path-{family}-{scenario_id}",
                        "component": "model_path",
                        "subject": family,
                        "phases": ["checkpoint"],
                        "work": {"checkpoint": work},
                        "resources": _resources(checkpoint, scenario, test_batch_size),
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
