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
import os
import re
import time
import urllib.error
import urllib.request
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


def _api(method: str, path: str, body: Optional[dict] = None) -> Any:
    """A GitHub API call with the token from the environment.

    Direct HTTP rather than the gh CLI: gh is not installed on every runner,
    and one urllib call removes a dependency the reporter should not need.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    req = urllib.request.Request(
        "https://api.github.com" + path,
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r) if r.length != 0 else None


def comment(pr: str, repo: str, body: Optional[str] = None) -> Optional[str]:
    """Find our comment by its marker, and write it when a body is given."""
    found = _api("GET", f"/repos/{repo}/issues/{pr}/comments?per_page=100")
    cid = next((c["id"] for c in (found or []) if marker(pr) in c["body"]), None)
    if body is None:
        return next((c["body"] for c in (found or []) if c["id"] == cid), None)
    if cid:
        _api("PATCH", f"/repos/{repo}/issues/comments/{cid}", {"body": body})
    else:
        _api("POST", f"/repos/{repo}/issues/{pr}/comments", {"body": body})
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
        except urllib.error.HTTPError:
            time.sleep(2 * (attempt + 1))  # another device wrote first, retry
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
