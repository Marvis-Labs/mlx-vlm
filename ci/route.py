"""Turn a diff into the set of cells that cover it.

Selection decides which architectures are involved. Expansion decides what
runs for each. The two change classes expand along different axes: a model
change fixes the architecture and varies the components, a component change
fixes the component and varies both the architectures and that component's
own configurations.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = Path(__file__).resolve().parent / "components"
MATRIX = ROOT / "mlx_vlm" / "tests" / "capabilities.json"
MODELS = Path(__file__).resolve().parent / "models.yaml"

# Weights are only part of the footprint; activations, the cache and the
# runtime add to it. This multiplier is a placeholder to be replaced by the
# peak_mem_gb that measurement actually reports.
OVERHEAD_FACTOR = 1.3
OVERHEAD_FIXED_GB = 4.0

# Devices reserve memory for the operating system; bootstrap.sh caps the
# wired limit at the same fraction. Sizing against raw capacity puts a
# 61 GB cell on a 64 GB device, which swaps or dies rather than measuring.
USABLE_FRACTION = 0.90


@dataclass
class Cell:
    id: str
    arch: str
    component: str
    config: str
    repo: str
    revision: str
    precision: str
    scenario: str
    regime: str
    requires_gb: float
    runs_on: List[str]
    metrics: List[str]
    args: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)


def load_components(*, enabled_only: bool = True) -> List[dict]:
    out = []
    for path in sorted(COMPONENTS.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        if enabled_only and not spec.get("enabled"):
            continue
        out.append(spec)
    return out


def fleet(repo: Optional[str]) -> List[int]:
    """Memory capacities advertised by registered runners, largest first.

    The router needs to know what hardware exists before it can decide which
    precision to emit. With no fleet reachable it returns nothing, and every
    cell is reported as unschedulable rather than silently dropped.
    """
    if not repo:
        return []
    try:
        raw = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/runners",
                "--jq",
                '.runners[] | select(.status=="online") | .labels[].name',
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except Exception:
        return []
    caps = {int(l[4:]) for l in raw.split() if l.startswith("mem-") and l[4:].isdigit()}
    return sorted(caps, reverse=True)


def signature(row: dict, columns: Sequence[str]) -> tuple:
    """Dedup key scoped to one component's columns.

    Scoping is load-bearing. Input modality says nothing about how an
    architecture exercises the cache implementation, so including it when
    routing a cache change multiplies representatives without adding
    coverage; excluding it when routing prefix caching would omit real
    behaviour, since prefix caching hashes media payloads.
    """
    key = []
    for col in columns:
        val = row.get(col)
        key.append(tuple(val) if isinstance(val, list) else val)
    return tuple(key)


def reaches(row: dict, spec: dict) -> bool:
    """Whether an architecture exercises this component at all."""
    for col in spec["columns"]:
        val = row.get(col)
        if col == "cache_kinds":
            continue  # every architecture has some cache
        if val:
            return True
    return False


def classify(paths: Sequence[str], specs: Sequence[dict]) -> dict:
    """Bucket changed paths into model, component and system changes."""
    out: Dict[str, Any] = {
        "models": set(),
        "components": [],
        "system": [],
        "unrouted": [],
    }
    comp_paths = {p: s for s in specs for p in s["paths"]}
    for path in paths:
        if path.startswith("mlx_vlm/models/") and path.count("/") >= 3:
            out["models"].add(path.split("/")[2])
        elif path in comp_paths:
            spec = comp_paths[path]
            if spec not in out["components"]:
                out["components"].append(spec)
        elif fnmatch.fnmatch(path, "mlx_vlm/server/*"):
            out["system"].append(path)
        else:
            out["unrouted"].append(path)
    return out


def pick_variant(arch: str, models: dict, budget_gb: Optional[float]) -> Optional[dict]:
    """Largest precision the fleet can hold — devices should be used fully.

    Falling back to a smaller precision keeps coverage rather than dropping
    the architecture, since precision is a property of the cell and not of
    the code under test.
    """
    variants = (models.get(arch) or {}).get("variants") or []
    ranked = sorted(variants, key=lambda v: v["weights_gb"], reverse=True)
    for v in ranked:
        need = v["weights_gb"] * OVERHEAD_FACTOR + OVERHEAD_FIXED_GB
        if budget_gb is None or need <= budget_gb:
            return {**v, "requires_gb": round(need, 1)}
    return None


def label_for(requires_gb: float, caps: Sequence[int]) -> Optional[List[str]]:
    fits = [c for c in sorted(caps) if c * USABLE_FRACTION >= requires_gb]
    return ["self-hosted", "macos", "arm64", f"mem-{fits[0]}"] if fits else None


def expand(arch: str, spec: dict, models: dict, caps: Sequence[int]) -> List[Cell]:
    budget = max(caps) * USABLE_FRACTION if caps else None
    variant = pick_variant(arch, models, budget)
    if variant is None:
        return []
    runs_on = label_for(variant["requires_gb"], caps)
    if runs_on is None:
        return []
    cells = []
    for regime in spec.get("regimes", ["single"]):
        for cfg in spec["configs"]:
            cells.append(
                Cell(
                    id=f"{arch}.{spec['name']}.{cfg['id']}.{variant['precision']}.{regime}",
                    arch=arch,
                    component=spec["name"],
                    config=cfg["id"],
                    repo=variant["repo"],
                    revision=variant.get("sha", ""),
                    precision=variant["precision"],
                    scenario=spec["scenario"],
                    regime=regime,
                    requires_gb=variant["requires_gb"],
                    runs_on=runs_on,
                    metrics=spec["metrics"],
                    args=cfg.get("args") or {},
                    env=cfg.get("env") or {},
                )
            )
    return cells


def route(paths: Sequence[str], gh_repo: Optional[str] = None) -> dict:
    matrix = json.loads(MATRIX.read_text())
    rows = matrix.get("architectures", matrix)
    models = yaml.safe_load(MODELS.read_text())
    specs = load_components()
    caps = fleet(gh_repo)
    buckets = classify(paths, specs)

    cells: Dict[str, Cell] = {}
    notes: List[str] = []

    # Model change: this architecture across every enabled component it
    # reaches. Actions asks for the model path; the router decides how far
    # the fleet can take it.
    for arch in sorted(buckets["models"]):
        if arch not in rows:
            notes.append(f"{arch}: not in the capability matrix")
            continue
        if arch not in models:
            notes.append(f"{arch}: no model metadata, cannot size a cell")
            continue
        for spec in specs:
            if not reaches(rows[arch], spec):
                continue
            produced = expand(arch, spec, models, caps)
            if not produced:
                notes.append(f"{arch}/{spec['name']}: no variant fits the fleet")
            for c in produced:
                cells[c.id] = c

    # Component change: one architecture per component-scoped signature,
    # across that component's configurations.
    for spec in buckets["components"]:
        pool = [a for a in sorted(rows) if reaches(rows[a], spec)]
        reps: Dict[tuple, str] = {}
        for arch in pool:
            reps.setdefault(signature(rows[arch], spec["columns"]), arch)
        for sig, arch in sorted(reps.items(), key=lambda kv: kv[1]):
            if arch not in models:
                # Substitute inside the signature class: members take an
                # identical path through the component, so this is free.
                alt = next(
                    (
                        a
                        for a in pool
                        if a in models and signature(rows[a], spec["columns"]) == sig
                    ),
                    None,
                )
                if alt is None:
                    notes.append(f"{spec['name']}: signature {sig} has no sized model")
                    continue
                arch = alt
            for c in expand(arch, spec, models, caps):
                cells[c.id] = c

    for path in buckets["unrouted"]:
        notes.append(f"unrouted: {path}")

    return {
        "cells": [asdict(c) for c in sorted(cells.values(), key=lambda c: c.id)],
        "fleet_gb": caps,
        "notes": notes,
    }


def changed_paths(base: str, head: str) -> List[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.splitlines() if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--paths", nargs="*", help="explicit paths instead of a diff")
    ap.add_argument(
        "--gh-repo", default=None, help="query this repo's runners for the fleet"
    )
    ap.add_argument(
        "--fleet", type=int, nargs="*", help="override fleet capacities in GB"
    )
    ap.add_argument("--out", help="append cells=<json> for GITHUB_OUTPUT")
    args = ap.parse_args()

    paths = args.paths or changed_paths(args.base, args.head)
    if args.fleet is not None:
        global fleet
        caps = sorted(args.fleet, reverse=True)
        fleet = lambda _r: caps  # noqa: E731
    result = route(paths, args.gh_repo)

    if args.out:
        with open(args.out, "a") as fh:
            fh.write("cells=" + json.dumps(result["cells"]) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
