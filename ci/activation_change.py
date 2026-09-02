from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ci.change_rules import ChangeContext, ChangeMatch


class SourceReader(Protocol):
    def read_text(self, revision: str, path: str) -> str: ...


class ActivationChange:
    """Plan independent contracts for shared activation profiles."""

    name = "activation_change"

    def __init__(self, config: Path, source: SourceReader):
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise ValueError(f"{config}: unsupported schema_version")
        source_path = data.get("source")
        configured = data.get("profiles")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"{config}: source must be a path")
        if not isinstance(configured, Mapping) or not configured:
            raise ValueError(f"{config}: profiles must be a non-empty mapping")
        profiles: dict[str, tuple[str, ...]] = {}
        downstream: dict[str, tuple[str, ...]] = {}
        for profile, value in configured.items():
            if not isinstance(profile, str) or not isinstance(value, Mapping):
                raise ValueError(f"{config}: invalid profile")
            symbols = value.get("symbols")
            consumers = value.get("downstream", [])
            if not isinstance(symbols, list) or any(
                not isinstance(symbol, str) or not symbol for symbol in symbols
            ):
                raise ValueError(f"{config}: invalid symbols for {profile}")
            if len(symbols) != len(set(symbols)):
                raise ValueError(f"{config}: duplicate symbols for {profile}")
            if not isinstance(consumers, list) or any(
                not isinstance(consumer, str) or not consumer for consumer in consumers
            ):
                raise ValueError(f"{config}: invalid downstream for {profile}")
            profiles[profile] = tuple(symbols)
            downstream[profile] = tuple(consumers)
        configured_symbols = [
            symbol for symbols in profiles.values() for symbol in symbols
        ]
        if len(configured_symbols) != len(set(configured_symbols)):
            raise ValueError(f"{config}: symbols must belong to one profile")
        self.source_path = source_path
        self.profiles = profiles
        self.downstream = downstream
        self.source = source

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        missing = [
            name
            for name, value in (
                ("base_sha", context.base_sha),
                ("head_sha", context.head_sha),
                ("contract_sha", context.target_sha),
            )
            if not value
        ]
        if missing:
            return {
                "component": self.name,
                "jobs": [],
                "gates": [],
                "blocked": [
                    {
                        "component": self.name,
                        "rule": self.name,
                        "changed_paths": paths,
                        "reason": "missing_immutable_revisions",
                        "missing": missing,
                    }
                ],
            }
        profiles, symbols, detection = self._changed_profiles(context)
        jobs = [
            {
                "id": f"activation_change:{profile}",
                "work_type": "ActivationChange",
                "component": self.name,
                "subject": profile,
                "profile": profile,
                "changed_paths": paths,
                "phases": ["activation_contract"],
                "required_memory_gib": 4,
                "required_disk_gib": 2,
                "base_sha": context.base_sha,
                "head_sha": context.head_sha,
                "contract_sha": context.target_sha,
                "activation_contract": {
                    "profile": profile,
                    "symbols": list(self.profiles[profile]),
                    "downstream": list(self.downstream[profile]),
                    "oracle": "independent_mathematical_contract",
                },
            }
            for profile in profiles
        ]
        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": [],
            "metadata": {
                "detection": detection,
                "symbols": symbols,
                "profiles": profiles,
            },
        }

    def _changed_profiles(
        self, context: ChangeContext
    ) -> tuple[list[str], list[str], str]:
        all_profiles = sorted(self.profiles)
        all_symbols = sorted(
            symbol for symbols in self.profiles.values() for symbol in symbols
        )
        if not context.base_sha or not context.head_sha:
            return all_profiles, all_symbols, "conservative_missing_revisions"
        try:
            base = self.source.read_text(context.base_sha, self.source_path)
            head = self.source.read_text(context.head_sha, self.source_path)
            base_symbols = self._symbol_bodies(base)
            head_symbols = self._symbol_bodies(head)
        except (OSError, subprocess.SubprocessError, SyntaxError, UnicodeError):
            return all_profiles, all_symbols, "conservative_source_error"
        configured = {
            symbol: profile
            for profile, symbols in self.profiles.items()
            for symbol in symbols
        }
        changed = sorted(
            symbol
            for symbol in configured
            if base_symbols.get(symbol) != head_symbols.get(symbol)
        )
        if changed:
            return (
                sorted({configured[symbol] for symbol in changed}),
                changed,
                "symbol_body_changed",
            )
        if ast.dump(ast.parse(base), include_attributes=False) != ast.dump(
            ast.parse(head), include_attributes=False
        ):
            return all_profiles, all_symbols, "conservative_module_change"
        return [], [], "no_semantic_change"

    @staticmethod
    def _symbol_bodies(source: str) -> dict[str, str]:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
