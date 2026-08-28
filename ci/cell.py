"""Run one cell against two revisions and report the delta.

The probe measures; this decides how many times and in what order, and it
owns the comparison. Two revisions of the same package cannot share a
process, so each measurement is a subprocess with PYTHONPATH pointed at a
worktree.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ci.verdict import FUNCTIONAL, LOWER_IS_BETTER, bar_for


def _probe_error(proc) -> str:
    """A consistent, human reason from a crashed probe.

    Every failure reads "category: detail". A crash reports the actual
    exception from the last traceback line -- expand_dims on a quantized
    cache, say -- rather than the useless "exited 1"; the exit code alone
    told the operator nothing.
    """
    tail = [l.strip() for l in proc.stderr.splitlines() if l.strip()]
    # The last line of a Python traceback is "ExcType: message".
    for line in reversed(tail):
        if ": " in line and line[0].isupper() and "Error" in line.split(":")[0]:
            kind, _, msg = line.partition(": ")
            return f"crashed: {msg[:80] or kind}"
    if "MemoryError" in proc.stderr or "metal" in proc.stderr.lower():
        return "crashed: out of memory"
    return f"crashed: probe exited {proc.returncode}"


def probe(
    worktree: Path,
    cell_path: Path,
    label: str,
    warmup: int,
    iterations: int,
    timeout: int,
) -> Dict[str, Any]:
    """One measurement inside one revision."""
    # The harness lives in the same repository as the code under test, so a
    # worktree carries its own copy of this package -- an older one, whose
    # probe may not exist or may not agree with this orchestrator. Invoking
    # the probe by absolute file path keeps sys.path[0] on the harness
    # directory, and PYTHONPATH supplies mlx_vlm from the revision.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree)
    env["CI_REVISION_LABEL"] = label
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "probe.py"),
                "--cell",
                str(cell_path),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            cwd=str(Path(cell_path).resolve().parent),
        )
    except subprocess.TimeoutExpired:
        # A hung load or a very large download that outruns the per-probe budget
        # must read as a clear timeout, not propagate uncaught and leave the
        # cell with the workflow's opaque "job failed before measuring".
        return {"error": f"timed out after {timeout}s (download or load too slow)"}
    if proc.returncode != 0:
        return {"error": _probe_error(proc)}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "error": "bad output: probe printed no result json",
            "stdout": proc.stdout[-800:],
            "stderr": proc.stderr[-800:],
        }


def interleave(
    base: Path, head: Path, cell_path: Path, *, repeats: int, warmup: int, timeout: int
) -> Dict[str, Any]:
    """Alternate revisions rather than running each in a block.

    Running all of A then all of B loads slow drift -- thermal state,
    background activity, cache warmth -- entirely onto the second half,
    where it is indistinguishable from a regression. Alternating cancels it.
    """
    runs: Dict[str, List[dict]] = {"base": [], "head": []}
    errors: List[str] = []
    for i in range(repeats):
        # Counterbalanced, not merely alternating. Running base then head
        # every time cancels drift across pairs but not bias within one: the
        # revision that always goes second inherits whatever the first leaves
        # behind. Measured on a mini, prefill for the second position was
        # pinned near the slow end while the first ranged over both, which
        # reads as a forty percent regression in code that changed neither.
        order = (("base", base), ("head", head))
        if i % 2:
            order = tuple(reversed(order))
        for label, tree in order:
            # Every repeat is a fresh subprocess with a cold Metal kernel
            # cache, so each one needs its own warmup. Warming only the
            # first leaves the rest measuring compilation.
            out = probe(tree, cell_path, label, warmup, 1, timeout)
            if "error" in out:
                # The reason is the signal; which repeat hit it is noise, and
                # every repeat usually fails the same way. Keep it once.
                if out["error"] not in errors:
                    errors.append(out["error"])
                continue
            runs[label].extend(out["runs"])
            runs.setdefault(f"{label}_env", []).append(
                {"mlx": out.get("mlx_version"), "python": out.get("python")}
            )
    return {"runs": runs, "errors": errors}


def summarize(samples: List[dict]) -> Dict[str, Any]:
    """Median per metric, with the spread that justifies a threshold."""
    if not samples:
        return {}
    out: Dict[str, Any] = {}
    numeric = [k for k in samples[0] if isinstance(samples[0][k], (int, float))]
    for key in numeric:
        vals = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        if not vals:
            continue
        med = statistics.median(vals)
        # Median absolute deviation, not standard deviation. With three or
        # four samples one slow iteration -- a background task, a first-touch
        # page fault -- inflates the standard deviation enough to make every
        # result inconclusive. Observed spread across identical cells ran
        # from two to sixty-five percent on the same device for that reason.
        # The scale factor makes it comparable to a standard deviation for
        # well-behaved data, while a single outlier cannot dominate it.
        mad = statistics.median([abs(v - med) for v in vals])
        sd = mad * 1.4826 if len(vals) > 1 else 0.0
        # Range grows with sample count and is therefore useless as a noise
        # floor: measuring more made it look worse. Coefficient of variation
        # is stable in n, and the standard error of the mean says how well
        # the centre is known, which is what a threshold must clear.
        out[key] = {
            "median": round(med, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "cv_pct": round(sd / med * 100, 2) if med else 0.0,
            "stderr_pct": round(sd / (len(vals) ** 0.5) / med * 100, 2) if med else 0.0,
            "n": len(vals),
        }
    hashes = {s.get("output_hash") for s in samples if s.get("output_hash")}
    if hashes:
        out["output_hash"] = sorted(hashes)
    # Non-numeric values are kept as the last observation rather than dropped.
    # APC reports its rejection reasons as a mapping, and losing it silently
    # made a declared metric look like one that never moved.
    for key, val in samples[-1].items():
        if key not in out and not isinstance(val, (int, float)):
            out[key] = val
    return out


def compare(base: Dict[str, Any], head: Dict[str, Any]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    for key, b in base.items():
        h = head.get(key)
        # A summary entry, not any mapping: keeping non-numeric metrics such
        # as the prefix-cache rejection reasons put a plain dict in here, and
        # treating it as a metric raised a KeyError mid-run.
        if not (
            isinstance(b, dict)
            and isinstance(h, dict)
            and "median" in b
            and "median" in h
        ):
            continue
        bm, hm = b["median"], h["median"]
        if bm == 0:
            # A counter that starts at zero has no percentage to report, but
            # dropping it hid the most interesting case: prefix reuse going
            # from none to some, or a functional path switching on.
            deltas[key] = {
                "base": bm,
                "head": hm,
                "change_pct": None,
                "note": "zero baseline" if hm == 0 else f"0 -> {hm}",
                "significant": hm != 0,
                "functional": key in FUNCTIONAL,
            }
            continue
        change = (hm - bm) / bm * 100
        # A malformed or older-format summary may lack the spread fields;
        # treat that as zero uncertainty rather than crashing the whole run.
        se = 2 * (b.get("stderr_pct", 0) ** 2 + h.get("stderr_pct", 0) ** 2) ** 0.5
        bar = bar_for(key, se)
        if key in LOWER_IS_BETTER:
            change = -change  # normalise so positive always means better
        deltas[key] = {
            "base": bm,
            "head": hm,
            "change_pct": round(change, 2),
            "cv_pct": round(max(b.get("cv_pct", 0), h.get("cv_pct", 0)), 2),
            # The bar a delta must clear: two standard errors of the
            # difference, or the smallest change worth acting on, whichever
            # is larger. Peak memory is perfectly repeatable, so its standard
            # error is zero and without the floor any nonzero delta passes.
            "noise_pct": round(bar, 2),
            "stderr_pct": round(se, 2),
            "significant": abs(change) > bar,
            "functional": key in FUNCTIONAL,
        }
    bh, hh = base.get("output_hash"), head.get("output_hash")
    if bh and hh:
        deltas["output_changed"] = bh != hh
    return deltas


def fingerprint() -> Dict[str, Any]:
    def sysctl(name: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "?"

    return {
        "host": platform.node(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30) or None,
        "macos": platform.mac_ver()[0],
        "wired_limit_mb": sysctl("iogpu.wired_limit_mb"),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def fetch_revisions(
    repo_url: str, base: str, head: str, work: Path
) -> tuple[Path, Path]:
    """Two revisions of the source, fetched as tarballs -- no git, no CLT.

    A revision of pure-Python mlx-vlm is just a directory of files, so each is
    downloaded from GitHub's archive endpoint and extracted once. This runs on
    a bare Mac that has only uv and Python, with no Command Line Tools, which
    is what lets any Apple Silicon machine be a runner out of the box.

    Both revisions share the one pinned environment; if their dependency files
    differ we would be measuring the dependency, so that is refused.
    """
    import tarfile
    import urllib.request

    work.mkdir(parents=True, exist_ok=True)
    trees = []
    for rev in (base, head):
        tree = work / f"rev-{rev}"
        if not (tree / "mlx_vlm").exists():
            tgz = work / f"{rev}.tar.gz"
            urllib.request.urlretrieve(f"{repo_url}/archive/{rev}.tar.gz", tgz)
            tmp = work / f"x-{rev}"
            with tarfile.open(tgz) as t:
                t.extractall(tmp)
            inner = next(tmp.iterdir())  # the archive wraps one top dir
            tree.exists() or inner.rename(tree)
            tgz.unlink(missing_ok=True)
        trees.append(tree)

    # One environment for both: a dependency change would make the comparison
    # measure the dependency, not the diff.
    for f in ("pyproject.toml", "requirements.txt", "uv.lock"):
        a, b = trees[0] / f, trees[1] / f
        if a.exists() and b.exists() and a.read_bytes() != b.read_bytes():
            raise SystemExit(f"{f} differs between revisions; needs two environments")
    return trees[0], trees[1]


def choose_variant(cell: dict) -> dict:
    """Pick the largest precision this device can hold, by sensing its own RAM.

    This is where zero-configuration lives: the router shipped every variant
    and made no assumption about hardware, so a mini picks 4-bit and a big
    machine picks bf16 of the same cell without anyone configuring either. The
    result of choosing on-device is that base and head, which run in the same
    job, always share a precision, so the comparison stays valid.
    """
    variants = cell.get("variants")
    if not variants:  # a resolved cell (already has repo) passes through
        return cell
    import subprocess

    mem_gb = (
        int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout
        )
        / 2**30
    )
    usable = mem_gb * float(os.environ.get("USABLE_FRACTION", "90")) / 100
    # variants arrive largest first; take the first that fits.
    for v in variants:
        if v["requires_gb"] <= usable:
            resolved = dict(
                cell,
                repo=v["repo"],
                revision=v["revision"],
                precision=v["precision"],
                requires_gb=v["requires_gb"],
            )
            # A parity variant also carries its transformers reference; keep it
            # so the parity cell can find the model to check against. Dropping it
            # here is why the gate had no way to run.
            if "ref" in v:
                resolved["ref"] = v["ref"]
            # Keep the gated flag so a gated checkpoint without a token can be
            # declined with a clear reason instead of a mid-run 401.
            if v.get("gated"):
                resolved["gated"] = True
            return resolved
    smallest = min(variants, key=lambda v: v["requires_gb"])
    raise SystemExit(
        f"device has {usable:.0f}GB usable, smallest variant needs "
        f"{smallest['requires_gb']}GB"
    )


def warm(cell: dict) -> None:
    """Download the pinned weights once, before anything is timed.

    Both halves read the same warm cache. Clearing between them would make
    the second half re-download and report network throughput as a
    regression.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=cell["repo"],
        revision=cell["revision"] or None,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.py",
            "*.jinja",
        ],
    )


def release(work: Path) -> None:
    """Always runs. Removes the extracted revisions; keeps the weight cache,
    which is revision-pinned and costly to re-download."""
    import shutil

    for tree in list(work.glob("rev-*")) + list(work.glob("x-*")):
        shutil.rmtree(tree, ignore_errors=True)


def parity_metrics(cell: dict) -> tuple:
    """Correctness of a new model against its transformers reference, as
    (metrics, errors). A new model has no prior revision to compare, so this
    replaces the before/after measurement. The heavy deps (torch, transformers)
    import only here, never on the performance path.
    """
    ref = cell.get("ref")
    if not ref:
        return {}, ["parity cell has no reference repo"]
    from ci.parity import compare

    try:
        return compare(cell["repo"], ref), []
    except Exception as exc:  # a missing reference class, a load failure
        return {}, [f"parity failed: {type(exc).__name__}: {exc}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--base", required=True, help="base revision")
    ap.add_argument("--head", required=True, help="head revision")
    ap.add_argument(
        "--repo-url",
        default="https://github.com/Marvis-Labs/mlx-vlm-ci",
        help="source repo to fetch revision tarballs from",
    )
    ap.add_argument("--work", default=None, help="scratch dir for worktrees")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="result.json")
    args = ap.parse_args()

    cell_path = Path(args.cell).resolve()
    cell = choose_variant(json.loads(cell_path.read_text()))
    cell_path.write_text(json.dumps(cell))  # the probe reads the resolved cell

    # A parity cell checks a new model against a reference rather than measuring
    # a before/after, so it never fetches two revisions or runs the probe. This
    # is the execution path the gate was missing: routing produced parity cells
    # and the report rendered them, but the orchestrator ran the perf probe.
    if cell.get("component") == "parity":
        metrics, errors = parity_metrics(cell)
        result = {
            "cell": {**cell, **metrics},
            "device": fingerprint(),
            "delta": {},
            "errors": errors,
            "ok": not errors,
        }
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(
            json.dumps({"cell": cell["id"], "ok": result["ok"], "errors": errors[:3]})
        )
        return 0 if result["ok"] else 1

    work = Path(args.work or os.environ.get("CI_WORK", Path.home() / "ci-work"))
    work.mkdir(parents=True, exist_ok=True)

    try:
        base_tree, head_tree = fetch_revisions(
            args.repo_url, args.base, args.head, work
        )
        warm(cell)
        gathered = interleave(
            base_tree,
            head_tree,
            cell_path,
            repeats=args.repeats,
            warmup=args.warmup,
            timeout=args.timeout,
        )
    finally:
        release(work)

    base = summarize(gathered["runs"]["base"])
    head = summarize(gathered["runs"]["head"])
    result = {
        "cell": cell,
        "device": fingerprint(),
        # The raw iterations, so a statistic can be reconsidered without
        # re-running on hardware. Recomputing the dispersion after changing
        # how it is measured was impossible from summaries alone.
        "samples": gathered["runs"],
        "base": base,
        "head": head,
        "delta": compare(base, head),
        "errors": gathered["errors"],
        "ok": not gathered["errors"] and bool(base) and bool(head),
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {"cell": cell["id"], "ok": result["ok"], "errors": gathered["errors"][:3]}
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
