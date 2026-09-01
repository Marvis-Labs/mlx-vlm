from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ci.change_rules import ChangeContext, ChangeMatch


class SourceReader(Protocol):
    def read_text(self, revision: str, path: str) -> str: ...


class KVCacheChange:
    """Plan trusted semantic contracts for shared KV-cache changes."""

    name = "kv_cache_change"

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
        contracts: dict[str, str] = {}
        for profile, value in configured.items():
            if not isinstance(profile, str) or not isinstance(value, Mapping):
                raise ValueError(f"{config}: invalid profile")
            implementations = value.get("implementations")
            contract = value.get("contract")
            if not isinstance(implementations, list) or any(
                not isinstance(symbol, str) or not symbol for symbol in implementations
            ):
                raise ValueError(f"{config}: invalid implementations for {profile}")
            if contract is not None and (
                not isinstance(contract, str) or ":" not in contract
            ):
                raise ValueError(f"{config}: invalid contract for {profile}")
            profiles[profile] = tuple(implementations)
            if contract is not None:
                contracts[profile] = contract
        self.source_path = source_path
        self.profiles = profiles
        self.contracts = contracts
        self.source = source

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        missing = [
            name
            for name, value in (
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
        unsupported = sorted(set(profiles) - self.contracts.keys())
        runnable = sorted(set(profiles) & self.contracts.keys())
        jobs = [
            {
                "id": f"kv_cache_change:{profile}",
                "work_type": "KVCacheChange",
                "component": self.name,
                "profile": profile,
                "changed_paths": paths,
                "phases": ["kv_cache_contract"],
                "minimum_memory_gib": 8,
                "required_disk_gib": 2,
                "head_sha": context.head_sha,
                "contract_sha": context.target_sha,
                "kv_cache_contract": {
                    "profile": profile,
                    "implementations": list(implementations),
                    "oracle": "independent_semantic_contract",
                    "entry_point": self.contracts[profile],
                },
            }
            for profile in runnable
            for implementations in (self.profiles[profile],)
        ]
        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": [
                {
                    "component": self.name,
                    "rule": self.name,
                    "changed_paths": paths,
                    "reason": "contract_profile_not_implemented",
                    "profile": profile,
                }
                for profile in unsupported
            ],
            "metadata": {
                "detection": detection,
                "symbols": symbols,
                "profiles": profiles,
            },
        }

    def _changed_profiles(
        self, context: ChangeContext
    ) -> tuple[list[str], list[str], str]:
        supported = sorted(self.contracts)
        if not context.base_sha or not context.head_sha:
            symbols = sorted(
                symbol for profile in supported for symbol in self.profiles[profile]
            )
            return supported, symbols, "conservative_missing_revisions"
        try:
            base = self.source.read_text(context.base_sha, self.source_path)
            head = self.source.read_text(context.head_sha, self.source_path)
            base_classes = self._class_bodies(base)
            head_classes = self._class_bodies(head)
        except (OSError, subprocess.SubprocessError, SyntaxError, UnicodeError):
            symbols = sorted(
                symbol for profile in supported for symbol in self.profiles[profile]
            )
            return supported, symbols, "conservative_source_error"

        configured = {
            symbol: profile
            for profile, symbols in self.profiles.items()
            for symbol in symbols
        }
        changed = sorted(
            symbol
            for symbol in configured
            if base_classes.get(symbol) != head_classes.get(symbol)
        )
        if changed:
            profiles = sorted({configured[symbol] for symbol in changed})
            return profiles, changed, "class_body_changed"
        if ast.dump(ast.parse(base), include_attributes=False) != ast.dump(
            ast.parse(head), include_attributes=False
        ):
            symbols = sorted(
                symbol for profile in supported for symbol in self.profiles[profile]
            )
            return supported, symbols, "conservative_module_change"
        return [], [], "no_semantic_change"

    @staticmethod
    def _class_bodies(source: str) -> dict[str, str]:
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)
        }
