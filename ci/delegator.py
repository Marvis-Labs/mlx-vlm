from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol, Sequence

import yaml


class ChangeComponent(Protocol):
    name: str

    def plan(self, changed_files: Iterable[str]) -> dict[str, Any]: ...


class ModelPath:
    """Build CI jobs for changes below mlx_vlm/models/<model>."""

    name = "model_path"

    def __init__(self, model_config: Path, scenario_config: Path):
        self.model_data = self._load_yaml(model_config)
        self.scenario_data = self._load_yaml(scenario_config)
        self.models = self._mapping(self.model_data, "models", model_config)
        self.profiles = self._mapping(
            self.model_data, "synthetic_profiles", model_config
        )
        self.scenarios = self._mapping(self.scenario_data, "scenarios", scenario_config)
        self.defaults_by_capability = self._mapping(
            self.scenario_data, "defaults_by_capability", scenario_config
        )
        self._require_schema_version(self.model_data, model_config)
        self._require_schema_version(self.scenario_data, scenario_config)

    def plan(self, changed_files: Iterable[str]) -> dict[str, Any]:
        changed_models = self._changed_models(changed_files)
        jobs: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []

        for model_name, paths in sorted(changed_models.items()):
            model = self.models.get(model_name)
            if not isinstance(model, dict):
                blocked.append(
                    self._blocker(model_name, paths, None, "missing_model_config")
                )
                continue

            scenario_ids, scenario_error = self._scenario_ids(model)

            synthetic = model.get("synthetic")
            synthetic_error = self._synthetic_error(synthetic)
            if synthetic_error:
                blocked.append(
                    self._blocker(model_name, paths, "synthetic", synthetic_error)
                )
            elif scenario_error:
                blocked.append(
                    self._blocker(model_name, paths, "synthetic", scenario_error)
                )
            else:
                jobs.append(
                    {
                        "id": f"model_path:{model_name}:synthetic",
                        "component": self.name,
                        "model": model_name,
                        "mode": "synthetic",
                        "changed_paths": paths,
                        "scenarios": scenario_ids,
                        "synthetic": {
                            "adapter": synthetic["adapter"],
                            "profile": synthetic["profile"],
                        },
                    }
                )

            checkpoint = model.get("hf_checkpoint")
            checkpoint_error = self._checkpoint_error(checkpoint)
            if checkpoint_error:
                blocked.append(
                    self._blocker(model_name, paths, "hf_checkpoint", checkpoint_error)
                )
            elif scenario_error:
                blocked.append(
                    self._blocker(model_name, paths, "hf_checkpoint", scenario_error)
                )
            else:
                jobs.append(
                    {
                        "id": f"model_path:{model_name}:hf_checkpoint",
                        "component": self.name,
                        "model": model_name,
                        "mode": "hf_checkpoint",
                        "changed_paths": paths,
                        "scenarios": scenario_ids,
                        "hf_checkpoint": {
                            "repo": checkpoint["repo"],
                            "revision": checkpoint["revision"],
                            "expected_model_type": checkpoint["expected_model_type"],
                            "weight": checkpoint["weight"],
                        },
                    }
                )

        return {
            "component": self.name,
            "models": sorted(changed_models),
            "jobs": jobs,
            "blocked": blocked,
        }

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return data

    @staticmethod
    def _mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
        value = data.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{source}: {key} must be a mapping")
        return value

    @staticmethod
    def _require_schema_version(data: dict[str, Any], source: Path) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"{source}: unsupported schema_version")

    @staticmethod
    def _model_name(path: str) -> str | None:
        normalized = PurePosixPath(path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            return None
        parts = normalized.parts
        if len(parts) < 4 or parts[:2] != ("mlx_vlm", "models"):
            return None
        return parts[2]

    def _changed_models(self, changed_files: Iterable[str]) -> dict[str, list[str]]:
        paths_by_model: dict[str, set[str]] = defaultdict(set)
        for changed_file in changed_files:
            model_name = self._model_name(changed_file)
            if model_name:
                paths_by_model[model_name].add(changed_file)
        return {model: sorted(paths) for model, paths in paths_by_model.items()}

    def _scenario_ids(self, model: dict[str, Any]) -> tuple[list[str], str | None]:
        configured = model.get("scenarios")
        if configured is None:
            configured = [
                self.defaults_by_capability.get(capability)
                for capability in model.get("capabilities", [])
            ]
        if not isinstance(configured, list):
            return [], "invalid_scenarios"
        if not configured:
            return [], "missing_scenarios"
        if any(not isinstance(item, str) for item in configured):
            return [], "invalid_scenarios"

        scenario_ids = list(dict.fromkeys(configured))
        if any(item not in self.scenarios for item in scenario_ids):
            return [], "unknown_scenario"
        return scenario_ids, None

    def _synthetic_error(self, synthetic: Any) -> str | None:
        if not isinstance(synthetic, dict) or synthetic.get("status") != "configured":
            return "not_configured"
        if not isinstance(synthetic.get("adapter"), str):
            return "invalid_synthetic_adapter"
        profile = synthetic.get("profile")
        if not isinstance(profile, str) or profile not in self.profiles:
            return "invalid_synthetic_profile"
        return None

    @staticmethod
    def _checkpoint_error(checkpoint: Any) -> str | None:
        if not isinstance(checkpoint, dict) or checkpoint.get("status") != "configured":
            return "not_configured"
        for key in ("repo", "revision", "expected_model_type"):
            if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
                return f"invalid_checkpoint_{key}"
        weight = checkpoint.get("weight")
        if not isinstance(weight, dict):
            return "invalid_checkpoint_weight"
        if not isinstance(weight.get("bytes"), int) or weight["bytes"] <= 0:
            return "invalid_checkpoint_weight_bytes"
        return None

    @staticmethod
    def _blocker(
        model: str, paths: list[str], mode: str | None, reason: str
    ) -> dict[str, Any]:
        return {
            "component": ModelPath.name,
            "model": model,
            "mode": mode,
            "changed_paths": paths,
            "reason": reason,
        }


class Delegator:
    """Run independent change components and merge their job plans."""

    def __init__(self, components: Sequence[ChangeComponent]):
        self.components = tuple(components)

    def plan(self, changed_files: Iterable[str]) -> dict[str, Any]:
        files = tuple(changed_files)
        component_plans = [component.plan(files) for component in self.components]
        active = [plan for plan in component_plans if plan["models"]]
        return {
            "schema_version": 1,
            "components": [plan["component"] for plan in active],
            "jobs": [job for plan in active for job in plan["jobs"]],
            "blocked": [item for plan in active for item in plan["blocked"]],
        }


def _resolve_commit(ref: str, cwd: Path | None) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_name_status(output: bytes) -> tuple[str, ...]:
    fields = output.rstrip(b"\0").split(b"\0") if output else []
    changed: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii")
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise ValueError("malformed git diff output")
        changed.extend(os.fsdecode(path) for path in fields[index : index + path_count])
        index += path_count
    return tuple(changed)


def changed_files_from_git(
    base: str, head: str, cwd: Path | None = None
) -> tuple[str, ...]:
    """Return changed paths between the merge base and head, including rename ends."""

    base_commit = _resolve_commit(base, cwd)
    head_commit = _resolve_commit(head, cwd)
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base_commit}...{head_commit}",
            "--",
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return _parse_name_status(result.stdout)


def default_delegator(config_directory: Path | None = None) -> Delegator:
    """Create the delegator using the repository model and scenario manifests."""

    config_directory = config_directory or Path(__file__).resolve().parent
    return Delegator(
        [
            ModelPath(
                config_directory / "model_path.yaml",
                config_directory / "model-path-scenario.yaml",
            )
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.changed_file and (args.base or args.head):
        parser.error("use --changed-file or --base/--head, not both")
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be provided together")
    if not args.changed_file and not args.base:
        parser.error("provide --changed-file or --base/--head")

    changed_files = (
        tuple(args.changed_file)
        if args.changed_file
        else changed_files_from_git(args.base, args.head)
    )
    output = json.dumps(default_delegator().plan(changed_files), indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
