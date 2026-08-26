"""One comment per pull request, updated as devices finish.

The comment is created before anything runs, listing every cell the router
selected as pending, and each cell patches its own row when it completes. A
cell that never reaches a device stays visibly unrun rather than vanishing:
"not tested" is a result, and a report that silently omits it reads as
coverage that did not happen.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

SHOWN = ["decode_tps", "prefill_tps", "ttft_ms", "peak_mem_gb"]

# A metric has to clear both statistical confidence and an amount worth
# acting on. Peak memory is perfectly repeatable, so its standard error is
# zero and without a floor any nonzero delta reads as significant.
FLOOR_PCT = {
    "peak_mem_gb": 2.0,
    "decode_tps": 2.0,
    "prefill_tps": 3.0,
    "ttft_ms": 5.0,
    "wall_ms": 2.0,
}
DEFAULT_FLOOR_PCT = 3.0
INCONCLUSIVE_RATIO = 2.0

STATUS = {
    "pending": "⏳ pending",
    "running": "🔄 running",
    "ok": "✅",
    "regressed": "🔴 regressed",
    "inconclusive": "⚠️ inconclusive",
    "failed": "❌ failed",
    "no-device": "⬜ no device",
}


def marker(pr: str) -> str:
    return f"<!-- mlx-vlm-ci:{pr} -->"


def verdict(metric: str, delta: dict) -> str:
    floor = FLOOR_PCT.get(metric, DEFAULT_FLOOR_PCT)
    noise = delta.get("noise_pct", 0)
    if noise > floor * INCONCLUSIVE_RATIO:
        return "inconclusive"
    # Judged here rather than read from the result: a run measured under an
    # earlier rule would otherwise carry that rule's verdict forward.
    if abs(delta["change_pct"]) <= max(noise, floor):
        return "noise"
    return "improved" if delta["change_pct"] > 0 else "regressed"


def row(
    cell_id: str,
    state: str,
    device: str = "—",
    deltas: Optional[dict] = None,
    note: str = "",
) -> str:
    cols = [f"`{cell_id}`", device, STATUS.get(state, state)]
    for k in SHOWN:
        v = (deltas or {}).get(k)
        if not isinstance(v, dict):
            cols.append("")
            continue
        state_k = verdict(k, v)
        mark = {"regressed": "🔴", "improved": "🟢", "noise": "", "inconclusive": "⚠️"}[
            state_k
        ]
        cols.append(f"{v['change_pct']:+.1f}% {mark}".strip())
    cols.append(note)
    return "| " + " | ".join(cols) + " |"


def header(pr: str, cells: int) -> list[str]:
    return [
        marker(pr),
        f"**mlx-vlm benchmark** — {cells} cells",
        "",
        "| cell | device | status | " + " | ".join(SHOWN) + " | note |",
        "|" + "---|" * (len(SHOWN) + 4),
    ]


def summarise(body: str) -> str:
    """Recompute the headline from the rows currently in the comment."""
    counts = {k: body.count(v) for k, v in STATUS.items()}
    if counts["regressed"]:
        head = f"{counts['regressed']} regression(s)"
    elif counts["pending"] or counts["running"]:
        done = (
            counts["ok"]
            + counts["regressed"]
            + counts["inconclusive"]
            + counts["failed"]
        )
        head = (
            f"running — {done} of {done + counts['pending'] + counts['running']} done"
        )
    elif counts["inconclusive"]:
        head = "inconclusive — device too noisy to decide"
    elif counts["failed"] or counts["no-device"]:
        head = "incomplete — some cells never ran"
    else:
        head = "no regression"
    return re.sub(
        r"\*\*mlx-vlm benchmark\*\* — [^\n]*",
        f"**mlx-vlm benchmark** — {head}",
        body,
        count=1,
    )


def find_comment(pr: str, repo: str) -> Optional[str]:
    out = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{pr}/comments",
            "--paginate",
            "--jq",
            f'.[] | select(.body | contains("{marker(pr)}")) | .id',
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    return out[0] if out else None


def read_comment(cid: str, repo: str) -> str:
    return json.loads(
        subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/comments/{cid}"],
            capture_output=True,
            text=True,
        ).stdout
    )["body"]


def write_comment(cid: Optional[str], pr: str, repo: str, body: str) -> None:
    if cid:
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{repo}/issues/comments/{cid}",
                "-f",
                f"body={body}",
            ],
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            ["gh", "pr", "comment", pr, "--repo", repo, "--body", body],
            check=True,
            capture_output=True,
        )


def cmd_init(args) -> int:
    """Post the matrix before anything runs, every cell pending."""
    cells = json.loads(Path(args.cells).read_text())
    unrouted = json.loads(Path(args.notes).read_text()) if args.notes else []
    lines = header(args.pr, len(cells))
    for c in sorted(cells, key=lambda c: c["id"]):
        lines.append(row(c["id"], "pending", device=c["runs_on"][-1]))
    for n in unrouted:
        lines.append(row(n, "no-device", note="no variant fits the fleet"))
    lines += [
        "",
        "<sub>Rows update as devices finish. A cell that never reaches a "
        "device stays marked, because not tested is not the same as passed.</sub>",
    ]
    body = summarise("\n".join(lines))
    write_comment(find_comment(args.pr, args.repo), args.pr, args.repo, body)
    return 0


def cmd_update(args) -> int:
    """Patch one cell's row in place, retrying if another device raced us."""
    result = json.loads(Path(args.result).read_text())
    cell_id = result["cell"]["id"]
    dev = result["device"]
    device = f"{dev['chip']} {dev['memory_gb']}GB"

    states = {
        verdict(k, v)
        for k, v in result["delta"].items()
        if isinstance(v, dict) and k in SHOWN
    }
    if not result.get("ok"):
        state, note = "failed", (result.get("errors") or [""])[0][:60]
    elif "regressed" in states:
        state, note = "regressed", ""
    elif "inconclusive" in states:
        state, note = "inconclusive", "re-run on an idle device"
    else:
        state, note = "ok", ""

    new = row(cell_id, state, device, result["delta"], note)
    for attempt in range(5):
        cid = find_comment(args.pr, args.repo)
        if cid is None:
            return 1
        body = read_comment(cid, args.repo)
        patched = re.sub(
            rf"^\| `{re.escape(cell_id)}` \|.*$",
            new.replace("\\", "\\\\"),
            body,
            count=1,
            flags=re.M,
        )
        if patched == body:  # our row was not there
            patched = body.rstrip() + "\n" + new
        try:
            write_comment(cid, args.pr, args.repo, summarise(patched))
            return 0
        except subprocess.CalledProcessError:
            time.sleep(2 * (attempt + 1))  # another device wrote first
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True)
    ap.add_argument("--repo", default="Blaizzy/mlx-vlm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--cells", required=True)
    i.add_argument("--notes")
    u = sub.add_parser("update")
    u.add_argument("--result", required=True)
    args = ap.parse_args()
    return cmd_init(args) if args.cmd == "init" else cmd_update(args)


if __name__ == "__main__":
    raise SystemExit(main())
