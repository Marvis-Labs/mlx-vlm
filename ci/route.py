"""Turn a diff into the set of cells that cover it.

Selection decides which architectures are involved. Expansion decides what
runs for each. The two change classes expand along different axes: a model
change fixes the architecture and varies the components, a component change
fixes the component and varies both the architectures and that component's
own configurations.
"""

from __future__ import annotations

import argparse
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

# Peak memory measured against weights on disk, gemma-2-2b-it-4bit:
#   184 tokens -> 1.39x    730 -> 1.99x    2940 -> 1.90x    11832 -> 2.76x
# Overhead is a roughly fixed activation cost plus a cache term that grows
# with context, so the multiplier belongs to the scenario rather than being
# one constant. The earlier "weights * 1.3 + 4 GB" was wrong in both
# directions: far too conservative for small models, optimistic for large
# ones, where the constant stops dominating.
SCENARIO_MULTIPLIER = {
    "single_generation": 2.0,
    "shared_prefix_pair": 2.0,
    "long_prompt": 2.8,
}
DEFAULT_MULTIPLIER = 2.8

# What the pinned measurement environment provides.
INSTALLED = {
    "mlx",
    "mlx_vlm",
    "huggingface_hub",
    "yaml",
}  # unknown scenario: assume the worst

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


def fleet(repo: Optional[str] = None) -> List[int]:
    """Runner memory sizes, largest first, read from a committed file.

    Not discovered from the API: listing self-hosted runners needs an admin
    token the workflow has no way to hold, and the fleet changes rarely. A
    committed list is simpler, needs no credentials, and shows up in review.
    """
    path = Path(__file__).resolve().parent / "fleet.txt"
    caps = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.isdigit():
            caps.append(int(line))
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
    """Whether an architecture supports this component at all.

    Separate from the signature scope: a column can distinguish how an
    architecture exercises a component without determining whether it can
    reach it. Prefix caching hashes media payloads, so modality belongs in
    its signature, but a model with neither form of reuse does not support
    prefix caching however many images it accepts.
    """
    required = spec.get("requires")
    if required is None:
        raise KeyError(f"{spec['name']}: declare `requires`, even if empty")
    return not required or any(row.get(col) for col in required)


def classify(paths: Sequence[str], specs: Sequence[dict]) -> dict:
    """Bucket changed paths into model, component and system changes."""
    out: Dict[str, Any] = {
        "models": set(),
        "components": [],
        "system": [],
        "unrouted": [],
    }
    comp_paths = {p: s for s in specs for p in s["paths"]}

    def behavioural(path: str) -> bool:
        # A test or a document inside a model directory does not change what
        # the model does, so it should not trigger a benchmark of it.
        name = path.rsplit("/", 1)[-1]
        return not (
            name.startswith("test_") or name.endswith(("_test.py", ".md", ".txt"))
        )

    for path in paths:
        if (
            path.startswith("mlx_vlm/models/")
            and path.count("/") >= 3
            and behavioural(path)
        ):
            out["models"].add(path.split("/")[2])
        elif path in comp_paths:
            spec = comp_paths[path]
            if spec not in out["components"]:
                out["components"].append(spec)
        elif path.startswith("mlx_vlm/server/"):
            out["system"].append(path)
        else:
            out["unrouted"].append(path)
    return out


def pick_variant(
    arch: str,
    models: dict,
    budget_gb: Optional[float],
    scenario: str = "single_generation",
) -> Optional[dict]:
    """Largest precision the fleet can hold — devices should be used fully.

    Falling back to a smaller precision keeps coverage rather than dropping
    the architecture, since precision is a property of the cell and not of
    the code under test.
    """
    entry = models.get(arch) or {}
    # Some checkpoints need packages the measurement environment does not
    # pin -- a remote-code processor that imports torch, for instance. The
    # cell would be dispatched, occupy a runner, and fail every time.
    missing = [p for p in entry.get("needs", []) if p not in INSTALLED]
    if missing:
        return None
    variants = entry.get("variants") or []
    ranked = sorted(variants, key=lambda v: v["weights_gb"], reverse=True)
    mult = SCENARIO_MULTIPLIER.get(scenario, DEFAULT_MULTIPLIER)
    for v in ranked:
        need = v["weights_gb"] * mult
        if budget_gb is None or need <= budget_gb:
            return {**v, "requires_gb": round(need, 1)}
    return None


# Must match the ladder a runner advertises in device.sh. A cell asks for the
# smallest tier that holds it, and every runner at or above that tier carries
# the matching label, so the cell is not pinned to one machine size.
TIERS = [16, 32, 48, 64, 96, 128, 192, 256, 512]


def label_for(requires_gb: float, caps: Sequence[int]) -> Optional[List[str]]:
    biggest = max(caps) * USABLE_FRACTION if caps else 0
    if requires_gb > biggest:
        return None
    tier = next((t for t in TIERS if t * USABLE_FRACTION >= requires_gb), None)
    return ["self-hosted", "macos", "arm64", f"mem-{tier}"] if tier else None


def expand(arch: str, spec: dict, models: dict, caps: Sequence[int]) -> List[Cell]:
    budget = max(caps) * USABLE_FRACTION if caps else None
    variant = pick_variant(arch, models, budget, spec["scenario"])
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


def route(
    paths: Sequence[str],
    caps: Optional[Sequence[int]] = None,
) -> dict:
    matrix = json.loads(MATRIX.read_text())
    rows = matrix.get("architectures", matrix)
    models = yaml.safe_load(MODELS.read_text())
    specs = load_components()
    caps = caps if caps is not None else fleet()
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
        "--fleet", type=int, nargs="*", help="override fleet capacities in GB"
    )
    ap.add_argument("--out", help="append cells=<json> for GITHUB_OUTPUT")
    ap.add_argument("--write", help="directory for cells.json and notes.json")
    args = ap.parse_args()

    paths = args.paths or changed_paths(args.base, args.head)
    caps = sorted(args.fleet, reverse=True) if args.fleet is not None else None
    result = route(paths, caps=caps)

    if args.out:
        with open(args.out, "a") as fh:
            fh.write("cells=" + json.dumps(result["cells"]) + "\n")
    if args.write:
        # The reporter needs these as files. Writing them here keeps the
        # workflow from unpacking the router's own output with a one-liner.
        Path(args.write, "cells.json").write_text(json.dumps(result["cells"]))
        Path(args.write, "notes.json").write_text(json.dumps(result["notes"]))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
