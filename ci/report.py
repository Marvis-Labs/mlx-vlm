"""Turn cell results into one comment on the pull request.

A delta is only reported as a regression when it exceeds the noise measured
on the device that produced it. The noise floor is a property of the machine,
not the metric: the same cell showed 15.74% variation in prefill throughput
on a busy laptop and 0.09% on an idle mini, so a single global threshold
would either miss regressions on quiet hardware or invent them on busy
hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Metrics worth reporting, in the order a reader wants them.
SHOWN = ["decode_tps", "prefill_tps", "ttft_ms", "peak_mem_gb"]


# When the noise bar is far wider than the change worth acting on, the device
# could not have detected a regression of that size. Reporting that as "no
# change" is the dangerous failure: an injected 7% prefill regression came
# back unflagged on a busy machine, and the report said all clear. Say
# inconclusive instead, so a widened bar is visible rather than reassuring.
INCONCLUSIVE_RATIO = 2.0
FLOOR_PCT = {
    "peak_mem_gb": 2.0,
    "decode_tps": 2.0,
    "prefill_tps": 3.0,
    "ttft_ms": 5.0,
    "wall_ms": 2.0,
}
DEFAULT_FLOOR_PCT = 3.0


def verdict(metric: str, delta: dict) -> str:
    """regressed / improved / noise / inconclusive, per device."""
    floor = FLOOR_PCT.get(metric, DEFAULT_FLOOR_PCT)
    noise = delta.get("noise_pct", 0)
    if noise > floor * INCONCLUSIVE_RATIO:
        return "inconclusive"
    # Judged here rather than trusting the stored flag: a result measured
    # under an earlier rule carries its verdict with it, and a report that
    # believes it would relabel old runs by whatever rule produced them.
    if abs(delta["change_pct"]) <= max(noise, floor):
        return "noise"
    return "improved" if delta["change_pct"] > 0 else "regressed"


def render(results: list[dict]) -> str:
    lines = []
    devices = {r["device"]["host"]: r["device"] for r in results}
    tally = {"regressed": 0, "inconclusive": 0}
    for r in results:
        for k, v in r["delta"].items():
            if isinstance(v, dict) and k in SHOWN:
                tally[verdict(k, v)] = tally.get(verdict(k, v), 0) + 1

    if tally["regressed"]:
        head = f"{tally['regressed']} regression(s)"
    elif tally["inconclusive"]:
        head = "inconclusive"
    else:
        head = "no regression"
    lines.append(
        f"**benchmark: {head}** — {len(results)} cells on "
        + ", ".join(f"{d['chip']} {d['memory_gb']}GB" for d in devices.values())
    )
    lines.append("")
    lines.append("| cell | device | " + " | ".join(SHOWN) + " | output |")
    lines.append("|" + "---|" * (len(SHOWN) + 3))

    for r in sorted(results, key=lambda r: (r["cell"]["id"], r["device"]["host"])):
        cells = [f"{r['device']['chip']} {r['device']['memory_gb']}GB"]
        for k in SHOWN:
            v = r["delta"].get(k)
            if not isinstance(v, dict):
                cells.append("—")
                continue
            state = verdict(k, v)
            if state == "inconclusive":
                cells.append(f"{v['change_pct']:+.1f}% ⚠️ ±{v['noise_pct']:.0f}%")
                continue
            mark = {"regressed": "🔴", "improved": "🟢", "noise": ""}[state]
            cells.append(f"{v['change_pct']:+.1f}% {mark}".strip())
        changed = r["delta"].get("output_changed")
        cells.append("changed" if changed else "same")
        lines.append(f"| `{r['cell']['id']}` | " + " | ".join(cells) + " |")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        lines.append("")
        lines.append("Cells that did not complete:")
        for r in failed:
            lines.append(f"- `{r['cell']['id']}`: {'; '.join(r['errors'][:2])}")

    lines.append("")
    if tally["inconclusive"]:
        lines.append(
            f"⚠️ {tally['inconclusive']} metric(s) inconclusive: the device was too "
            "noisy to detect a change worth acting on. Re-run on an idle device."
        )
        lines.append("")
    lines.append(
        "<sub>Percentages are normalised so positive is better. 🔴 and 🟢 mark "
        "changes exceeding both two standard errors on that device and a minimum "
        "worth acting on. ⚠️ means the noise bar was wider than that minimum, so "
        "no conclusion is possible either way.</sub>"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory of result json files")
    ap.add_argument("--pr", help="pull request number; prints instead when absent")
    ap.add_argument("--upstream", default="Blaizzy/mlx-vlm")
    args = ap.parse_args()

    results = []
    for path in sorted(Path(args.results).rglob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except Exception:
            continue
    if not results:
        print("no results to report")
        return 0

    body = render(results)
    if not args.pr:
        print(body)
        return 0
    # Posting is an explicit step, never a side effect of collecting results.
    subprocess.run(
        ["gh", "pr", "comment", args.pr, "--repo", args.upstream, "--body", body],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
