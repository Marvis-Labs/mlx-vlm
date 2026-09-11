from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_ci.repository.change_rules import ChangeContext, ChangeMatch, load_yaml_mapping


class ModelPath:
    """Build CI jobs for configured existing-model changes."""

    name = "model_path"

    def __init__(
        self,
        model_config: Path,
        scenario_config: Path,
        supported_synthetic_adapters: frozenset[str] | None = None,
    ):
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
        self.supported_synthetic_adapters = supported_synthetic_adapters
        self._require_schema_version(self.model_data, model_config)
        self._require_schema_version(self.scenario_data, scenario_config)

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        changed_models, invalid = self._changed_models(matches)
        jobs: list[dict[str, Any]] = []
        blocked = list(invalid)

        if {"ci/config/models.yaml", "ci/config/scenarios.yaml"} & set(
            context.changed_files
        ):
            blocked.extend(
                self._blocker(
                    model_name,
                    paths,
                    None,
                    "existing_model_ci_configuration_changed",
                )
                for model_name, paths in sorted(changed_models.items())
            )
            return {
                "component": self.name,
                "jobs": [],
                "gates": [],
                "blocked": blocked,
            }

        for model_name, paths in sorted(changed_models.items()):
            model = self.models.get(model_name)
            if not isinstance(model, dict):
                blocked.append(
                    self._blocker(model_name, paths, None, "missing_model_config")
                )
                continue

            configuration, configuration_blockers = self.configuration(
                model_name, paths
            )
            blocked.extend(configuration_blockers)
            if configuration is not None:
                jobs.append(self.work_item(model_name, paths, configuration))

        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": blocked,
        }

    def work_item(
        self,
        model_name: str,
        paths: list[str],
        configuration: Mapping[str, Any],
        *,
        component: str | None = None,
    ) -> dict[str, Any]:
        from ci.model_path.component import resource_requirements

        checkpoint = dict(configuration["hf_checkpoint"])
        required_memory, required_disk = resource_requirements(checkpoint)
        return {
            "id": f"model_path:{model_name}",
            "work_type": "ModelPath",
            "component": component or self.name,
            "subject": model_name,
            "model": model_name,
            "changed_paths": paths,
            "phases": ["synthetic", "hf_checkpoint"],
            "scenarios": list(configuration["scenarios"]),
            "synthetic": dict(configuration["synthetic"]),
            "hf_checkpoint": checkpoint,
            "required_memory_gib": required_memory,
            "required_disk_gib": required_disk,
        }

    def configuration(
        self, model_name: str, paths: list[str]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        model = self.models.get(model_name)
        if not isinstance(model, dict):
            return None, [
                self._blocker(model_name, paths, None, "missing_model_config")
            ]

        scenario_ids, scenario_error = self._scenario_ids(model)
        synthetic = model.get("synthetic")
        checkpoint = model.get("hf_checkpoint")
        errors = (
            ("synthetic", self._synthetic_error(synthetic) or scenario_error),
            (
                "hf_checkpoint",
                self._checkpoint_error(checkpoint) or scenario_error,
            ),
        )
        blockers = [
            self._blocker(model_name, paths, mode, reason)
            for mode, reason in errors
            if reason
        ]
        if blockers:
            return None, blockers
        return {
            "synthetic": {
                "adapter": synthetic["adapter"],
                "profile": synthetic["profile"],
            },
            "hf_checkpoint": {
                "repo": checkpoint["repo"],
                "revision": checkpoint["revision"],
                "expected_model_type": checkpoint["expected_model_type"],
                "weight": checkpoint["weight"],
            },
            "scenarios": scenario_ids,
        }, []

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        return load_yaml_mapping(path)

    @staticmethod
    def _mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
        value = data.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{source}: {key} must be a mapping")  # noqa: TRY004
        return value

    @staticmethod
    def _require_schema_version(data: dict[str, Any], source: Path) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"{source}: unsupported schema_version")

    @staticmethod
    def _changed_models(
        matches: Sequence[ChangeMatch],
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
        paths_by_model: dict[str, set[str]] = defaultdict(set)
        blocked: list[dict[str, Any]] = []
        for match in matches:
            model_name = match.captures.get("model")
            if not model_name:
                blocked.append(
                    {
                        "component": match.component,
                        "rule": match.rule,
                        "changed_paths": [match.path],
                        "reason": "missing_model_capture",
                    }
                )
                continue
            paths_by_model[model_name].add(match.path)
        return (
            {model: sorted(paths) for model, paths in paths_by_model.items()},
            blocked,
        )

    def _scenario_ids(self, model: dict[str, Any]) -> tuple[list[str], str | None]:
        configured = model.get("scenarios")
        if configured is None:
            capabilities = model.get("capabilities", [])
            if not isinstance(capabilities, list):
                return [], "invalid_capabilities"
            configured = [
                self.defaults_by_capability.get(capability)
                for capability in capabilities
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
        if not isinstance(synthetic.get("adapter"), str) or not synthetic["adapter"]:
            return "invalid_synthetic_adapter"
        if (
            self.supported_synthetic_adapters is not None
            and synthetic["adapter"] not in self.supported_synthetic_adapters
        ):
            return "unsupported_synthetic_adapter"
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
        from mlx_ci.repository.checkpoint_policy import (
            CheckpointPolicyError,
            validate_checkpoint,
        )

        try:
            validate_checkpoint(
                {
                    "repo": checkpoint["repo"],
                    "revision": checkpoint["revision"],
                    "expected_model_type": checkpoint["expected_model_type"],
                    "weight": checkpoint["weight"],
                }
            )
        except CheckpointPolicyError:
            return "unsafe_checkpoint_configuration"
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


class NewModelPath:
    """Require approval before executing contributor-supplied new-model tests."""

    name = "new_model_path"

    def __init__(self, model_path: ModelPath):
        self.model_path = model_path

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        changed_models, invalid = self.model_path._changed_models(matches)
        gates: list[dict[str, Any]] = []
        blocked = list(invalid)

        for model_name, paths in sorted(changed_models.items()):
            if "ci/config/models.yaml" not in context.changed_files:
                blocked.append(
                    {
                        "component": self.name,
                        "model": model_name,
                        "mode": None,
                        "changed_paths": paths,
                        "reason": "model_manifest_not_updated",
                    }
                )
                continue
            configuration, configuration_blockers = self.model_path.configuration(
                model_name, paths
            )
            blocked.extend(
                {**blocker, "component": self.name}
                for blocker in configuration_blockers
            )
            if configuration is None:
                continue
            if not context.head_sha:
                blocked.append(
                    {
                        "component": self.name,
                        "model": model_name,
                        "mode": None,
                        "changed_paths": paths,
                        "reason": "missing_head_sha",
                    }
                )
                continue
            configuration_digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        configuration, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            )
            gates.append(
                {
                    "id": f"new_model_path:{model_name}:{context.head_sha}",
                    "type": "maintainer_approval",
                    "status": "awaiting_maintainer_approval",
                    "component": self.name,
                    "model": model_name,
                    "head_sha": context.head_sha,
                    "configuration_digest": configuration_digest,
                    "changed_paths": paths,
                    "requested_phases": ["synthetic", "hf_checkpoint"],
                    "configuration": configuration,
                    "pending_work": self.model_path.work_item(
                        model_name,
                        paths,
                        configuration,
                    ),
                }
            )

        return {
            "component": self.name,
            "jobs": [],
            "gates": gates,
            "blocked": blocked,
        }
