"""Render the benchmark comment. Runs on the cloud runner, never on a device.

The comment is posted once as pending when routing finishes, then edited once
to the final table when every cell has reported. A device only ever writes a
result file; it never touches the comment, which is what lets a runner carry
no token, no gh, and no copy of this code.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ci.verdict import FUNCTIONAL, verdict

SPEED = ["decode_tps", "prefill_tps", "ttft_ms", "peak_mem_gb"]
COLUMNS = SPEED + sorted(FUNCTIONAL)

STATUS = {
    "pending": "⏳",
    "ok": "✅",
    "regressed": "🔴",
    "improved": "🟢",
    "inconclusive": "⚠️",
    "failed": "❌",
    "no-device": "⬜",
}


def marker(pr: str) -> str:
    return f"<!-- mlx-vlm-ci:{pr} -->"


def _cell_state(delta: dict) -> str:
    states = {
        verdict(k, v) for k, v in delta.items() if isinstance(v, dict) and k in COLUMNS
    }
    for s in ("regressed", "inconclusive"):
        if s in states:
            return s
    return "ok"


def _row(cell_id: str, state: str, device: str, delta: dict, note: str) -> str:
    cells = [f"`{cell_id}`", device, STATUS.get(state, state)]
    for k in COLUMNS:
        v = delta.get(k)
        if not isinstance(v, dict):
            cells.append("")
        elif v.get("change_pct") is None:
            cells.append(v.get("note", "—"))
        else:
            mark = {
                "regressed": "🔴",
                "improved": "🟢",
                "noise": "",
                "inconclusive": "⚠️",
            }
            cells.append(f"{v['change_pct']:+.1f}% {mark[verdict(k, v)]}".strip())
    cells.append(note)
    return "| " + " | ".join(cells) + " |"


def render(pr: str, cells: list, results: Optional[list] = None) -> str:
    by_id = {r["cell"]["id"]: r for r in (results or [])}
    rows, regressed, pending = [], 0, 0
    for c in sorted(cells, key=lambda c: c["id"]):
        r = by_id.get(c["id"])
        if r is None:
            rows.append(_row(c["id"], "pending", c["runs_on"][-1], {}, ""))
            pending += 1
            continue
        dev = f"{r['device']['chip']} {r['device']['memory_gb']}GB"
        if not r.get("ok"):
            rows.append(
                _row(c["id"], "failed", dev, {}, (r.get("errors") or [""])[0][:48])
            )
            continue
        state = _cell_state(r["delta"])
        regressed += state == "regressed"
        rows.append(
            _row(
                c["id"],
                state,
                dev,
                r["delta"],
                "re-run on an idle device" if state == "inconclusive" else "",
            )
        )

    if pending:
        head = f"running — {len(cells) - pending} of {len(cells)} done"
    elif regressed:
        head = f"{regressed} regression(s)"
    else:
        head = "no regression"

    lines = [
        marker(pr),
        f"**mlx-vlm benchmark** — {head}",
        "",
        "| cell | device | " + " | ".join(["status"] + COLUMNS) + " | note |",
        "|" + "---|" * (len(COLUMNS) + 4),
    ]
    lines += rows
    lines += [
        "",
        "<sub>Positive is better. 🔴/🟢 mark changes past both two standard "
        "errors and a floor worth acting on; ⚠️ means the device was too noisy "
        "to decide.</sub>",
    ]
    return "\n".join(lines)


def _api(method: str, path: str, body: Optional[dict] = None) -> Any:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    req = urllib.request.Request(
        "https://api.github.com" + path,
        method=method,
        data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r) if r.length else None


def upsert(pr: str, repo: str, body: str) -> None:
    """Post the comment, or edit ours if it already exists. Only the cloud
    job calls this, one write at a time, so there is no race to guard."""
    found = _api("GET", f"/repos/{repo}/issues/{pr}/comments?per_page=100") or []
    cid = next((c["id"] for c in found if marker(pr) in c["body"]), None)
    if cid:
        _api("PATCH", f"/repos/{repo}/issues/comments/{cid}", {"body": body})
    else:
        _api("POST", f"/repos/{repo}/issues/{pr}/comments", {"body": body})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cells", required=True, help="the routed cell list")
    ap.add_argument("--results", help="directory of result json; omit for pending")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells).read_text())
    results = None
    if args.results:
        results = [
            json.loads(p.read_text()) for p in Path(args.results).rglob("*.json")
        ]
    upsert(args.pr, args.repo, render(args.pr, cells, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
