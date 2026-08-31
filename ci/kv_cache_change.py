from __future__ import annotations

from typing import Any, Sequence

from ci.change_rules import ChangeContext, ChangeMatch


class KVCacheChange:
    """Plan trusted semantic contracts for shared KV-cache changes."""

    name = "kv_cache_change"
    profiles = {
        "dense": ("KVCache", "SimpleKVCache"),
    }

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
                },
            }
            for profile, implementations in sorted(self.profiles.items())
        ]
        return {
            "component": self.name,
            "jobs": jobs,
            "gates": [],
            "blocked": [],
        }
