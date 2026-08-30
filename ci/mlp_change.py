from __future__ import annotations

import ast
import io
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ci.change_rules import ChangeContext, ChangeMatch


class SourceReader(Protocol):
    def read_text(self, revision: str, path: str) -> str: ...


class GitSource:
    def __init__(self, repository: Path):
        self.repository = repository

    def read_text(self, revision: str, path: str) -> str:
        result = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def read_many(self, revision: str, paths: Sequence[str]) -> dict[str, str]:
        if not paths:
            return {}
        result = subprocess.run(
            ["git", "archive", "--format=tar", revision, "--", *paths],
            cwd=self.repository,
            check=True,
            capture_output=True,
        )
        values: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                stream = archive.extractfile(member)
                if stream is not None:
                    values[member.name] = stream.read().decode()
        return values


class MLPChange:
    """Plan shared MLP validation from immutable base and head source trees."""

    name = "mlp_change"

    def __init__(self, config: Path, model_path: Any, source: SourceReader):
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise ValueError(f"{config}: unsupported schema_version")
        source_path = data.get("source")
        symbols = data.get("symbols")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"{config}: source must be a path")
        if not isinstance(symbols, Mapping) or not symbols:
            raise ValueError(f"{config}: symbols must be a non-empty mapping")
        self.source_path = source_path
        self.coverage = self._coverage(symbols, config)
        self.model_path = model_path
        self.source = source

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        symbols, detection = self._changed_symbols(context)
        consumers = self._consumers(context)
        validation_errors = self._validate_consumers(consumers)
        if validation_errors:
            return {
                "component": self.name,
                "jobs": [],
                "gates": [],
                "blocked": [
                    {
                        "component": self.name,
                        "rule": "mlp_change",
                        "changed_paths": paths,
                        "reason": reason,
                    }
                    for reason in validation_errors
                ],
            }

        origins_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
        for symbol in symbols:
            for model in self.coverage[symbol]:
                origins_by_model[model].append(
                    {"change_type": "MLPChange", "symbol": symbol}
                )

        jobs = [
            self._work_item(model, origins, paths, consumers)
            for model, origins in sorted(origins_by_model.items())
        ]
        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": [],
            "metadata": {
                "detection": detection,
                "symbols": symbols,
            },
        }

    def _changed_symbols(self, context: ChangeContext) -> tuple[list[str], str]:
        all_symbols = list(self.coverage)
        if not context.base_sha or not context.head_sha:
            return all_symbols, "conservative_missing_revisions"
        try:
            base = self.source.read_text(context.base_sha, self.source_path)
            head = self.source.read_text(context.head_sha, self.source_path)
            base_classes = self._class_bodies(base)
            head_classes = self._class_bodies(head)
        except (OSError, subprocess.SubprocessError, SyntaxError, UnicodeError):
            return all_symbols, "conservative_source_error"
        if set(base_classes) != set(all_symbols) or set(head_classes) != set(
            all_symbols
        ):
            return all_symbols, "conservative_class_set_changed"
        changed = [
            symbol
            for symbol in all_symbols
            if base_classes[symbol] != head_classes[symbol]
        ]
        if changed:
            return changed, "class_body_changed"
        if ast.dump(ast.parse(base), include_attributes=False) != ast.dump(
            ast.parse(head), include_attributes=False
        ):
            return all_symbols, "conservative_module_change"
        return [], "no_semantic_change"

    def _consumers(self, context: ChangeContext) -> dict[str, set[str]]:
        consumers = {symbol: set() for symbol in self.coverage}
        if not context.base_sha or not context.head_sha:
            for symbol, families in self.coverage.items():
                consumers[symbol].update(families)
            return consumers
        revisions = (
            (context.base_sha, context.base_files),
            (context.head_sha, context.head_files),
        )
        for revision, files in revisions:
            candidates = sorted(
                path
                for path in files
                if path.startswith("mlx_vlm/models/")
                and path.endswith(".py")
                and path != self.source_path
            )
            read_many = getattr(self.source, "read_many", None)
            values = (
                read_many(revision, candidates)
                if callable(read_many)
                else {
                    path: self.source.read_text(revision, path) for path in candidates
                }
            )
            for path, source in values.items():
                try:
                    tree = ast.parse(source)
                except (OSError, subprocess.SubprocessError, SyntaxError, UnicodeError):
                    continue
                imported = self._imported_symbols(tree)
                family = Path(path).parts[2]
                for symbol in imported & self.coverage.keys():
                    consumers[symbol].add(family)
        return consumers

    def _validate_consumers(self, consumers: Mapping[str, set[str]]) -> list[str]:
        errors: list[str] = []
        for symbol, selected in self.coverage.items():
            actual = consumers.get(symbol, set())
            missing = sorted(set(selected) - actual)
            if missing:
                errors.append(f"mlp_manifest_non_consumer:{symbol}:{','.join(missing)}")
            if symbol != "SwiGLUMLP" and set(selected) != actual:
                omitted = sorted(actual - set(selected))
                if omitted:
                    errors.append(
                        f"mlp_manifest_missing_consumer:{symbol}:{','.join(omitted)}"
                    )
        return errors

    def _work_item(
        self,
        model: str,
        origins: list[dict[str, str]],
        paths: list[str],
        consumers: Mapping[str, set[str]],
    ) -> dict[str, Any]:
        symbols = [origin["symbol"] for origin in origins]
        job: dict[str, Any] = {
            "id": f"model_path:{model}",
            "work_type": "ModelPath",
            "component": "model_path",
            "model": model,
            "changed_paths": paths,
            "origins": origins,
            "phases": ["mlp_contract"],
            "minimum_memory_gib": 8,
            "required_disk_gib": 2,
            "mlp_contract": {
                "symbols": symbols,
                "consumer": model,
                "consumer_verified": all(
                    model in consumers[symbol] for symbol in symbols
                ),
            },
        }
        checkpoint, reason = self._checkpoint(model)
        if checkpoint is not None:
            job["phases"].append("hf_checkpoint")
            job["hf_checkpoint"] = checkpoint["hf_checkpoint"]
            job["scenarios"] = checkpoint["scenarios"]
            job.pop("minimum_memory_gib")
            job.pop("required_disk_gib")
        else:
            job["unavailable_phases"] = {"hf_checkpoint": reason}
        return job

    def _checkpoint(self, model: str) -> tuple[dict[str, Any] | None, str]:
        value = self.model_path.models.get(model)
        if not isinstance(value, dict):
            return None, "missing_model_config"
        scenario_ids, scenario_error = self.model_path._scenario_ids(value)
        checkpoint = value.get("hf_checkpoint")
        error = self.model_path._checkpoint_error(checkpoint) or scenario_error
        if error:
            return None, error
        return {
            "hf_checkpoint": {
                "repo": checkpoint["repo"],
                "revision": checkpoint["revision"],
                "expected_model_type": checkpoint["expected_model_type"],
                "weight": checkpoint["weight"],
            },
            "scenarios": scenario_ids,
        }, ""

    @staticmethod
    def _coverage(symbols: Mapping[str, Any], source: Path) -> dict[str, list[str]]:
        coverage: dict[str, list[str]] = {}
        for symbol, value in symbols.items():
            if not isinstance(symbol, str) or not isinstance(value, Mapping):
                raise ValueError(f"{source}: invalid symbol entry")
            families = value.get("families")
            if not isinstance(families, list) or any(
                not isinstance(family, str) or not family for family in families
            ):
                raise ValueError(f"{source}: {symbol} families must be strings")
            if len(families) != len(set(families)):
                raise ValueError(f"{source}: {symbol} families must be unique")
            expected = value.get("expected_families")
            if not isinstance(expected, int) or expected != len(families):
                raise ValueError(f"{source}: {symbol} family count does not match")
            coverage[symbol] = list(families)
        return coverage

    @staticmethod
    def _class_bodies(source: str) -> dict[str, str]:
        tree = ast.parse(source)
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

    @staticmethod
    def _imported_symbols(tree: ast.AST) -> set[str]:
        return {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith("mlp")
            for alias in node.names
        }
