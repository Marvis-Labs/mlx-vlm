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


def _diverge(
    change: float, significant: bool, half: int = 10, cap: float = 15.0
) -> str:
    """A bar from a center line: left is worse, right is better, capped."""
    n = min(round(abs(change) / cap * half), half) if significant else 0
    if n == 0:
        return "·" * half + "┃" + "·" * half
    if change < 0:
        return "·" * (half - n) + "█" * n + "┃" + "·" * half
    return "·" * half + "┃" + "█" * n + "·" * (half - n)


def _graph(cells: list, results: list) -> list:
    """A compact before/after graph, shown when one architecture is changed.

    A model-path change touches a single architecture across many configs, so
    the median change per metric is the headline; the diverging bar makes a
    regression visible at a glance without reading the table.
    """
    import statistics

    by_id = {r["cell"]["id"]: r for r in results}
    arch = cells[0]["arch"]
    per_metric: dict = {}
    for c in cells:
        r = by_id.get(c["id"])
        if not r or not r.get("ok"):
            continue
        for m, v in r["delta"].items():
            if isinstance(v, dict) and m in SPEED and v.get("change_pct") is not None:
                per_metric.setdefault(m, []).append((v["change_pct"], v["significant"]))
    if not per_metric:
        return []
    lines = [
        f"### `{arch}` — median change across configs",
        "",
        "```",
        f"{'':<13}worse ◄──────────┃──────────► better",
    ]
    for m in SPEED:
        pts = per_metric.get(m)
        if not pts:
            continue
        med = statistics.median([c for c, _ in pts])
        sig = sum(s for _, s in pts) > len(pts) / 2
        # A median past the metric's floor, in the majority of configs, is a
        # real regression; mark it so the headline can count it.
        from ci.verdict import floor_for

        mark = (
            " 🔴"
            if sig and med < -floor_for(m)
            else (" 🟢" if sig and med > floor_for(m) else "")
        )
        lines.append(f"{m:<13}{_diverge(med, sig)} {med:+6.1f}%{mark}")
    lines += ["```", ""]
    return lines


def _tier(cell: dict) -> str:
    labels = cell.get("runs_on") or []
    return labels[-1] if labels else "?"


def render(
    pr: str, cells: list, results: Optional[list] = None, notes: Optional[list] = None
) -> str:
    if not cells:
        # A change that reaches no runnable cell must say why -- an unsized
        # model, a disabled component, a path nothing maps to -- rather than
        # leaving a comment stuck at zero of zero.
        why = "; ".join(notes or []) or "no benchmarkable change detected"
        return "\n".join(
            [
                marker(pr),
                "**mlx-vlm benchmark** — nothing to run",
                "",
                f"<sub>{why}</sub>",
            ]
        )
    by_id = {r["cell"]["id"]: r for r in (results or [])}
    rows, regressed, pending = [], 0, 0
    for c in sorted(cells, key=lambda c: c["id"]):
        r = by_id.get(c["id"])
        if r is None:
            rows.append(_row(c["id"], "pending", _tier(c), {}, ""))
            pending += 1
            continue
        d = r.get("device") or {}
        dev = f"{d.get('chip', '?')} {d.get('memory_gb', 0)}GB"
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

    graph = []
    if results and len({c["arch"] for c in cells}) == 1:
        graph = _graph(cells, results)

    # For a single-architecture change, headline the shape of the change, not
    # a single cell. A real model regression moves a metric across most of its
    # configs; one config moving alone on a noisy device is noise the graph
    # already shows as flat, so the headline should agree with the graph
    # rather than counting a lone cell as a regression.
    if graph:
        regressed = sum(1 for line in graph if line.strip().endswith("🔴"))

    if pending:
        head = f"running — {len(cells) - pending} of {len(cells)} done"
    elif regressed:
        head = f"{regressed} regression(s)"
    else:
        head = "no regression"

    lines = [marker(pr), f"**mlx-vlm benchmark** — {head}", ""]
    lines += graph
    lines += [
        "<details><summary>per-cell detail</summary>",
        "",
        "| cell | device | " + " | ".join(["status"] + COLUMNS) + " | note |",
        "|" + "---|" * (len(COLUMNS) + 4),
    ]
    lines += rows
    lines += ["", "</details>"]
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
    ap.add_argument("--notes", help="routing notes, for the empty case")
    ap.add_argument("--results", help="directory of result json; omit for pending")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells).read_text())
    results = None
    if args.results:
        results = [
            json.loads(p.read_text()) for p in Path(args.results).rglob("result.json")
        ]
    upsert(args.pr, args.repo, render(args.pr, cells, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
