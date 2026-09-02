from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from ci.change_rules import ChangeContext, ChangeMatch, PathPattern


class SourceReader(Protocol):
    def read_text(self, revision: str, path: str) -> str: ...


class SecurityChange:
    """Apply trusted static security profiles to changed source."""

    name = "security_change"

    def __init__(self, config: Path, source: SourceReader):
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise ValueError(f"{config}: unsupported schema_version")
        configured = data.get("profiles")
        if not isinstance(configured, Mapping) or not configured:
            raise ValueError(f"{config}: profiles must be a non-empty mapping")
        self.profiles: dict[str, tuple[tuple[PathPattern, ...], frozenset[str]]] = {}
        for name, value in configured.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                raise ValueError(f"{config}: invalid security profile")
            paths = value.get("include")
            categories = value.get("categories")
            if not isinstance(paths, list) or not paths:
                raise ValueError(f"{config}: {name} requires include paths")
            if not isinstance(categories, list) or not categories:
                raise ValueError(f"{config}: {name} requires categories")
            self.profiles[name] = (
                tuple(PathPattern(str(path)) for path in paths),
                frozenset(str(category) for category in categories),
            )
        self.source = source

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        if not context.base_sha or not context.head_sha:
            return {
                "component": self.name,
                "jobs": [],
                "gates": [],
                "checks": [],
                "blocked": [
                    {
                        "component": self.name,
                        "reason": "missing_immutable_revisions",
                        "changed_paths": paths,
                    }
                ],
            }
        profile_paths: dict[str, list[str]] = defaultdict(list)
        violations: dict[str, list[dict[str, str]]] = defaultdict(list)
        errors: list[dict[str, Any]] = []
        for path in paths:
            selected = self._profiles(path)
            if not selected:
                continue
            for profile in selected:
                profile_paths[profile].append(path)
            if not path.endswith(".py"):
                continue
            try:
                base = (
                    self.source.read_text(context.base_sha, path)
                    if context.base_contains(path)
                    else ""
                )
            except Exception:
                base = ""
            try:
                head = (
                    self.source.read_text(context.head_sha, path)
                    if context.head_contains(path)
                    else ""
                )
                added = _new_findings(base, head)
            except (OSError, SyntaxError, UnicodeError) as error:
                errors.append(
                    {
                        "component": self.name,
                        "reason": "security_scan_failed",
                        "changed_paths": [path],
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            for profile in selected:
                categories = self.profiles[profile][1]
                violations[profile].extend(
                    finding for finding in added if finding["category"] in categories
                )
        checks = [
            {
                "component": self.name,
                "profile": profile,
                "subject": profile,
                "status": "passed" if not violations[profile] else "blocked",
                "changed_paths": sorted(profile_paths[profile]),
                "findings": violations[profile],
            }
            for profile in sorted(profile_paths)
        ]
        blockers = [
            {
                "component": self.name,
                "reason": "security_policy_violation",
                "profile": profile,
                "changed_paths": sorted(profile_paths[profile]),
                "findings": findings,
            }
            for profile, findings in sorted(violations.items())
            if findings
        ]
        return {
            "component": self.name,
            "jobs": [],
            "gates": [],
            "checks": checks,
            "blocked": errors + blockers,
        }

    def _profiles(self, path: str) -> tuple[str, ...]:
        return tuple(
            name
            for name, (patterns, _) in self.profiles.items()
            if any(pattern.match(path) is not None for pattern in patterns)
        )


def _new_findings(base: str, head: str) -> list[dict[str, str]]:
    before = Counter(_findings(base))
    after = Counter(_findings(head))
    return [
        {"category": category, "rule": rule}
        for (category, rule), count in sorted((after - before).items())
        for _ in range(count)
    ]


def _findings(source: str) -> list[tuple[str, str]]:
    if not source:
        return []
    tree = ast.parse(source)
    findings: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        keywords = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
        }
        if _is_true(keywords.get("trust_remote_code")):
            findings.append(("remote_code", "trust_remote_code_true"))
        if name == "torch.load" and not _is_true(keywords.get("weights_only")):
            findings.append(
                ("unsafe_deserialization", "torch_load_without_weights_only")
            )
        if name in {"pickle.load", "pickle.loads", "dill.load", "joblib.load"}:
            findings.append(("unsafe_deserialization", name.replace(".", "_")))
        if name in {"eval", "exec", "os.system"}:
            findings.append(("command_execution", name.replace(".", "_")))
        if name in {
            "subprocess.call",
            "subprocess.Popen",
            "subprocess.run",
        } and _is_true(keywords.get("shell")):
            findings.append(("command_execution", "subprocess_shell_true"))
        if name.endswith(".exec_module"):
            findings.append(("dynamic_code", "importlib_exec_module"))
    for name in ("HF_TOKEN", "GITHUB_TOKEN", "RUNNER_TOKEN"):
        if name in source:
            findings.append(("secret_access", f"access_{name.lower()}"))
    return findings


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True
