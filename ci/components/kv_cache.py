from __future__ import annotations

import sys
from typing import Any

from ci.components.base import (
    ComponentContext,
    ComponentRegistration,
    ExecutionContext,
    PhaseRegistration,
)


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.kv_cache_change import KVCacheChange
    from ci.mlp_change import GitSource

    return (
        KVCacheChange(
            context.config("components/kv_cache_profiles.yaml"),
            GitSource(context.repository),
        ),
    )


def _output():
    from ci.bot import KVCacheChangeOutput

    return KVCacheChangeOutput()


def _contract(context: ExecutionContext) -> list[str]:
    directory = context.config_directory
    return [
        sys.executable,
        str(directory / "kv_cache_contract_compare.py"),
        "--job",
        str(context.job_path),
        "--control",
        str(context.control),
        "--head",
        str(context.head),
        "--probe",
        str(directory / "kv_cache_contract_probe.py"),
    ]


REGISTRATION = ComponentRegistration(
    name="kv_cache_change",
    components=frozenset({"kv_cache_change"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset({("KVCacheChange", "kv_cache_change")}),
    phases=(PhaseRegistration("kv_cache_contract", _contract),),
)
