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

from ci.verdict import FUNCTIONAL, verdict

SHOWN = ["decode_tps", "prefill_tps", "ttft_ms", "peak_mem_gb"]

COLUMNS = SHOWN + sorted(FUNCTIONAL)


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


def row(
    cell_id: str,
    state: str,
    device: str = "—",
    deltas: Optional[dict] = None,
    note: str = "",
) -> str:
    deltas = deltas or {}
    cols = [f"`{cell_id}`", device, STATUS.get(state, state)]
    for k in COLUMNS:
        v = deltas.get(k)
        if not isinstance(v, dict):
            cols.append("")
            continue
        if v.get("change_pct") is None:  # counter with a zero baseline
            cols.append(v.get("note", "—"))
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
        "| cell | device | status | " + " | ".join(COLUMNS) + " | note |",
        "|" + "---|" * (len(COLUMNS) + 4),
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


def comment(pr: str, repo: str, body: Optional[str] = None) -> Optional[str]:
    """Find our comment, and write it when a body is given.

    One helper rather than three: locating, reading and writing were separate
    calls that were never used apart, and the split made the retry in an
    update read like more than a second attempt at the same thing.
    """
    ids = subprocess.run(
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
    cid = ids[0] if ids else None
    if body is None:
        if cid is None:
            return None
        return json.loads(
            subprocess.run(
                ["gh", "api", f"repos/{repo}/issues/comments/{cid}"],
                capture_output=True,
                text=True,
            ).stdout
        )["body"]
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
    return cid


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
    comment(args.pr, args.repo, body)
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
        body = comment(args.pr, args.repo)
        if body is None:
            return 1
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
            comment(args.pr, args.repo, summarise(patched))
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
