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

# Lower is better for these; everything else is higher-is-better.
LOWER_IS_BETTER = {"ttft_ms", "wall_ms", "peak_mem_gb"}

# Metrics that describe behaviour rather than speed. A regression here is a
# functional failure regardless of what the timings say.
FUNCTIONAL = {"token_hit_rate", "matched_tokens", "exact_hits"}

# A perfectly repeatable metric drives the standard error to zero, and then
# any nonzero delta clears the bar: peak memory was reported as a significant
# change at 0.40%. Statistical confidence is not the same as mattering, so a
# metric also has to move by an amount someone would act on.
FLOOR_PCT = {
    "peak_mem_gb": 2.0,
    "decode_tps": 2.0,
    "prefill_tps": 3.0,
    "ttft_ms": 5.0,
    "wall_ms": 2.0,
}
DEFAULT_FLOOR_PCT = 3.0


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
    if proc.returncode != 0:
        return {
            "error": f"probe exited {proc.returncode}",
            "stderr": proc.stderr[-2000:],
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "error": f"unparseable probe output: {exc}",
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
        for label, tree in (("base", base), ("head", head)):
            # Every repeat is a fresh subprocess with a cold Metal kernel
            # cache, so each one needs its own warmup. Warming only the
            # first leaves the rest measuring compilation.
            out = probe(tree, cell_path, label, warmup, 1, timeout)
            if "error" in out:
                errors.append(f"{label} repeat {i}: {out['error']}")
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
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
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
        if not isinstance(b, dict) or not isinstance(h, dict):
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
        se = 2 * (b["stderr_pct"] ** 2 + h["stderr_pct"] ** 2) ** 0.5
        bar = max(se, FLOOR_PCT.get(key, DEFAULT_FLOOR_PCT))
        if key in LOWER_IS_BETTER:
            change = -change  # normalise so positive always means better
        deltas[key] = {
            "base": bm,
            "head": hm,
            "change_pct": round(change, 2),
            "cv_pct": round(max(b["cv_pct"], h["cv_pct"]), 2),
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


def worktrees(repo: Path, base: str, head: str, work: Path) -> tuple[Path, Path]:
    """Two checkouts, one environment.

    mlx-vlm is pure Python, so a revision is a path. Sharing one pinned
    environment is what makes the comparison a comparison: if dependencies
    differed between the halves we would be measuring the dependency.
    """
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            base,
            head,
            "--",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "uv.lock",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed:
        raise SystemExit(
            f"dependency files differ between revisions:\n{changed}\n"
            "this cell needs two environments, not one"
        )
    # A worktree deleted from disk stays registered, and a re-add at the
    # same path then fails. Prune first so a wiped scratch dir recovers.
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True)
    out = []
    for rev in (base, head):
        tree = work / f"rev-{rev}"
        if not tree.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "add",
                    "-q",
                    "--detach",
                    str(tree),
                    rev,
                ],
                check=True,
            )
        out.append(tree)
    return out[0], out[1]


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


def release(repo: Path, work: Path) -> None:
    """Always runs. Keeps the weight cache: it is revision-pinned, and
    re-downloading it is pure cost."""
    quiesce = Path(__file__).resolve().parent / "runner" / "quiesce.sh"
    if quiesce.exists():
        subprocess.run([str(quiesce), "release"], capture_output=True)
    for tree in work.glob("rev-*"):
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(tree)],
            capture_output=True,
        )
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--base", required=True, help="base revision")
    ap.add_argument("--head", required=True, help="head revision")
    ap.add_argument("--repo", default=".", help="git checkout to make worktrees from")
    ap.add_argument("--work", default=None, help="scratch dir for worktrees")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", action="store_true", help="leave worktrees in place")
    ap.add_argument("--out", default="result.json")
    args = ap.parse_args()

    cell_path = Path(args.cell).resolve()
    cell = json.loads(cell_path.read_text())
    repo = Path(args.repo).resolve()
    work = Path(args.work or os.environ.get("CI_WORK", Path.home() / "ci-work"))
    work.mkdir(parents=True, exist_ok=True)

    try:
        base_tree, head_tree = worktrees(repo, args.base, args.head, work)
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
        if not args.keep:
            release(repo, work)

    base = summarize(gathered["runs"]["base"])
    head = summarize(gathered["runs"]["head"])
    result = {
        "cell": cell,
        "device": fingerprint(),
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
