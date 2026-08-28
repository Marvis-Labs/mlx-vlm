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
PARITY_MODELS = Path(__file__).resolve().parent / "parity_models.yaml"
MODELS = Path(__file__).resolve().parent / "models.yaml"
PROTECTED = Path(__file__).resolve().parent / "protected_paths.yaml"


def _protected(paths: Sequence[str]) -> tuple:
    """Split changed paths into (refuse, warn) against protected_paths.yaml.

    This list, like the rest of the harness, is read from the default branch,
    so a pull request cannot weaken it for its own run -- and editing the list
    is itself a refused change.
    """
    rules = yaml.safe_load(PROTECTED.read_text()) if PROTECTED.exists() else {}

    def match(path: str, pattern: str) -> bool:
        return path == pattern or (pattern.endswith("/") and path.startswith(pattern))

    refuse = sorted(
        p for p in paths if any(match(p, x) for x in rules.get("refuse") or [])
    )
    warn = sorted(
        p
        for p in paths
        if p not in refuse and any(match(p, x) for x in rules.get("warn") or [])
    )
    return refuse, warn


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
    scenario: str
    regime: str
    # Every precision the model is published in. The device that picks the
    # cell chooses the largest that fits its own memory, so the router never
    # needs to know what hardware exists.
    variants: List[Dict[str, Any]]
    min_gb: float  # memory for the smallest variant, the entry bar
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


def variants_for(arch: str, models: dict, scenario: str) -> List[Dict[str, Any]]:
    """Every publishable precision, each with the memory it needs, large first.

    The device picks from this at run time, so the router makes no assumption
    about the fleet. A checkpoint needing packages the measurement environment
    does not carry is dropped, since it would fail on every device.
    """
    entry = models.get(arch) or {}
    if [p for p in entry.get("needs", []) if p not in INSTALLED]:
        return []
    mult = SCENARIO_MULTIPLIER.get(scenario, DEFAULT_MULTIPLIER)
    out = []
    for v in sorted(
        entry.get("variants") or [], key=lambda v: v["weights_gb"], reverse=True
    ):
        # Prefer a size hardware actually measured over the estimate.
        measured = _measured_peak(arch, v["precision"])
        out.append(
            {
                "repo": v["repo"],
                "revision": v.get("sha", ""),
                "precision": v["precision"],
                "requires_gb": measured or round(v["weights_gb"] * mult, 1),
            }
        )
    return out


def _measured_peak(arch: str, precision: str):
    """The measured peak for (arch, precision), if the store has one.

    Best-effort: a missing database just falls back to the estimate."""
    try:
        from ci.store import measured_peak

        return measured_peak(arch, precision)
    except Exception:
        return None


# The ladder a runner advertises in device.sh. A cell is labelled with the tier
# for its smallest variant, so any device large enough to run it at some
# precision matches; that device then picks the largest precision it can hold.
TIERS = [16, 32, 48, 64, 96, 128, 192, 256, 512]


def label_for(min_gb: float) -> Optional[List[str]]:
    tier = next((t for t in TIERS if t * USABLE_FRACTION >= min_gb), None)
    return ["self-hosted", "macos", "arm64", f"mem-{tier}"] if tier else None


def expand(arch: str, spec: dict, models: dict) -> List[Cell]:
    variants = variants_for(arch, models, spec["scenario"])
    if not variants:
        return []
    min_gb = min(v["requires_gb"] for v in variants)  # smallest precision
    runs_on = label_for(min_gb)
    if runs_on is None:
        return []
    cells = []
    for regime in spec.get("regimes", ["single"]):
        for cfg in spec["configs"]:
            cells.append(
                Cell(
                    id=f"{arch}.{spec['name']}.{cfg['id']}.{regime}",
                    arch=arch,
                    component=spec["name"],
                    config=cfg["id"],
                    scenario=spec["scenario"],
                    regime=regime,
                    variants=variants,
                    min_gb=min_gb,
                    runs_on=runs_on,
                    metrics=spec["metrics"],
                    args=cfg.get("args") or {},
                    env=cfg.get("env") or {},
                )
            )
    return cells


def parity_cell(arch: str, pair: dict) -> Cell:
    """A correctness cell: check a new model against its reference. Labelled
    parity, since it needs a device with room for two copies and the
    transformers library the benchmark environment does not carry."""
    return Cell(
        id=f"{arch}.parity",
        arch=arch,
        component="parity",
        config="reference",
        scenario="parity",
        regime="single",
        variants=[
            {
                "repo": pair["mlx"],
                "revision": "",
                "precision": "bf16",
                "requires_gb": 0.0,
                "ref": pair["ref"],
            }
        ],
        min_gb=0.0,
        runs_on=["self-hosted", "macos", "arm64", "parity"],
        metrics=["greedy_agreement", "kl_mean", "kl_max"],
        args={},
        env={},
    )


def route(paths: Sequence[str]) -> dict:
    # A change to the CI's own security boundary is handled before anything is
    # measured. Refused files stop the run outright; warned files let it proceed
    # but flag the diff so a maintainer approves the change knowingly.
    refuse, warn = _protected(paths)
    if refuse:
        return {
            "cells": [],
            "notes": [
                "REFUSED: this pull request changes protected CI files ("
                + ", ".join(refuse)
                + "). The benchmark will not run on it. These define how the CI "
                "is triggered and what it is allowed to do, so they can only be "
                "changed by the owner committing directly to the default branch "
                "on GitHub -- never approved by a run on a pull request."
            ],
        }

    matrix = json.loads(MATRIX.read_text())
    rows = matrix.get("architectures", matrix)
    models = yaml.safe_load(MODELS.read_text())
    specs = load_components()
    buckets = classify(paths, specs)

    cells: Dict[str, Cell] = {}
    notes: List[str] = []

    # Model change: this architecture across every enabled component it
    # reaches. Actions asks for the model path; the router decides how far
    # the fleet can take it.
    parity = yaml.safe_load(PARITY_MODELS.read_text()) or {}
    for arch in sorted(buckets["models"]):
        if arch not in rows:
            # A model the matrix has never seen is new: there is no previous
            # revision to compare against, so it goes to the correctness gate
            # rather than the performance path.
            if arch in parity:
                c = parity_cell(arch, parity[arch])
                cells[c.id] = c
            else:
                notes.append(
                    f"{arch}: new model; add it to ci/parity_models.yaml "
                    f"to check it against a reference"
                )
            continue
        if arch not in models:
            notes.append(f"{arch}: no model metadata, cannot size a cell")
            continue
        for spec in specs:
            if not reaches(rows[arch], spec):
                continue
            produced = expand(arch, spec, models)
            if not produced:
                notes.append(f"{arch}/{spec['name']}: no runnable variant")
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
            for c in expand(arch, spec, models):
                cells[c.id] = c

    for path in buckets["unrouted"]:
        notes.append(f"unrouted: {path}")

    if warn:
        notes.insert(
            0,
            "WARNING: this pull request modifies CI harness files ("
            + ", ".join(warn)
            + "). It changes how the benchmark itself behaves -- review these "
            "before approving the run, not just the numbers below.",
        )

    return {
        "cells": [asdict(c) for c in sorted(cells.values(), key=lambda c: c.id)],
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
    ap.add_argument("--out", help="append cells=<json> for GITHUB_OUTPUT")
    ap.add_argument("--write", help="directory for cells.json and notes.json")
    args = ap.parse_args()

    paths = args.paths or changed_paths(args.base, args.head)
    result = route(paths)

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
