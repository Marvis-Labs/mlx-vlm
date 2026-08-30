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
    "busy": "⊘",
    "no-device": "⬜",
}


def marker(pr: str) -> str:
    return f"<!-- mlx-vlm:ci:{pr} -->"


def _cell_state(delta: dict) -> str:
    states = {
        verdict(k, v) for k, v in delta.items() if isinstance(v, dict) and k in COLUMNS
    }
    for s in ("regressed", "inconclusive"):
        if s in states:
            return s
    return "ok"


def _row(cell_id: str, state: str, device: str, delta: dict, note: str) -> str:
    # Show only the speed metrics: a wide table with the sparse functional
    # columns scrolled off-screen. Functional metrics still shape the status via
    # _cell_state, and a real problem lands in the note, so nothing is lost.
    cells = [f"`{cell_id}`", device, STATUS.get(state, state)]
    for k in SPEED:
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


METRIC_LABEL = {
    "decode_tps": "decode (tok/s)",
    "prefill_tps": "prefill (tok/s)",
    "ttft_ms": "time to first token",
    "peak_mem_gb": "peak memory",
}


def _summary(cells: list, results: list) -> tuple:
    """The headline table: median change per metric across every config, drawn
    the same from pending to done. Returns the lines and the count of metrics
    whose median is a real regression, so the headline agrees with the table.
    """
    import statistics

    from ci.verdict import floor_for

    by_id = {r["cell"]["id"]: r for r in results}
    per_metric: dict = {}
    for c in cells:
        r = by_id.get(c["id"])
        if not r or not r.get("ok"):
            continue
        for m, v in r["delta"].items():
            if isinstance(v, dict) and m in SPEED and v.get("change_pct") is not None:
                per_metric.setdefault(m, []).append((v["change_pct"], v["significant"]))

    arch = cells[0]["arch"]
    lines = [
        f"**`{arch}` — median change across configs** · positive is better",
        "",
        "| metric | change | |",
        "|:--|--:|:-:|",
    ]
    regressions = 0
    for m in SPEED:
        label = METRIC_LABEL.get(m, m)
        pts = per_metric.get(m)
        if not pts:
            lines.append(f"| {label} | — | ⏳ |")
            continue
        med = statistics.median([c for c, _ in pts])
        sig = sum(s for _, s in pts) > len(pts) / 2
        mark = ""
        if sig and med < -floor_for(m):
            mark, regressions = "🔴", regressions + 1
        elif sig and med > floor_for(m):
            mark = "🟢"
        lines.append(f"| {label} | {med:+.1f}% | {mark} |")
    lines += [""]
    return lines, regressions


_PARITY_THRESHOLDS: Optional[dict] = None


def _parity_threshold(arch: str) -> dict:
    """The reviewed parity tolerances for an architecture, falling back to the
    default. Loaded from the committed parity_thresholds.yaml so the gate
    applies the baseline on disk rather than a number buried in the renderer --
    which hardcoded greedy>=0.98 and ignored the KL ceilings entirely."""
    global _PARITY_THRESHOLDS
    if _PARITY_THRESHOLDS is None:
        import yaml

        p = Path(__file__).resolve().parent / "parity_thresholds.yaml"
        _PARITY_THRESHOLDS = yaml.safe_load(p.read_text()) if p.exists() else {}
    t = dict(_PARITY_THRESHOLDS.get("default", {}))
    t.update(_PARITY_THRESHOLDS.get(arch, {}))
    return t


def _parity_row(cell_id: str, device: str, res: dict) -> str:
    arch = res.get("arch") or cell_id.split(".")[0]
    t = _parity_threshold(arch)
    g, km, kx = (
        res.get("greedy_agreement"),
        res.get("kl_mean"),
        res.get("kl_max"),
    )
    # A parity cell passes only if it agrees enough AND diverges little enough:
    # greedy above its floor, both KL figures under their ceilings. Any breach
    # is a reported drift, not a silent pass, and the failing metric is flagged.
    g_bad = g is None or ("greedy" in t and g < t["greedy"])
    km_bad = km is not None and "kl_mean" in t and km > t["kl_mean"]
    kx_bad = kx is not None and "kl_max" in t and kx > t["kl_max"]
    mark = "🔴" if (g_bad or km_bad or kx_bad) else "✅"

    def flag(label: str, val, bad: bool) -> str:
        shown = f"{label} {val}" if val is not None else f"{label} —"
        return shown + (" 🔴" if bad else "")

    cols = [
        f"`{cell_id}`",
        device,
        mark,
        flag("greedy", f"{g:.3f}" if g is not None else None, g_bad),
        flag("kl_mean", km, km_bad),
        flag("kl_max", kx, kx_bad),
    ]
    return "| " + " | ".join(cols) + " |"


def _tier(cell: dict) -> str:
    labels = cell.get("runs_on") or []
    return labels[-1] if labels else "?"


def render(
    pr: str, cells: list, results: Optional[list] = None, notes: Optional[list] = None
) -> str:
    """One layout, identical from pending to done. A model-path change leads
    with the graph (bars pending until results arrive); a component change
    leads with the table. Failures always appear the same way: a red cross in
    the status column with the reason in the note column."""
    notes = notes or []
    # A change to the CI's own security boundary is surfaced above everything
    # else: a refusal replaces the report, a warning banners it.
    refusal = next((n for n in notes if n.startswith("REFUSED:")), None)
    if refusal:
        return "\n".join(
            [
                marker(pr),
                "**mlx-vlm benchmark** — ⛔ refused: protected CI files changed",
                "",
                f"> {refusal}",
            ]
        )
    warning = next((n for n in notes if n.startswith("WARNING:")), None)

    if not cells:
        why = "; ".join(notes) or "no benchmarkable change detected"
        return "\n".join(
            [
                marker(pr),
                "**mlx-vlm benchmark** — nothing to run",
                "",
                f"<sub>{why}</sub>",
            ]
        )

    results = results or []
    by_id = {r["cell"]["id"]: r for r in results}
    one_arch = len({c["arch"] for c in cells}) == 1

    ok = failed = pending = regressed = declined = 0
    rows = []
    for c in sorted(cells, key=lambda c: c["id"]):
        r = by_id.get(c["id"])
        if r is None:
            rows.append(_row(c["id"], "pending", _tier(c), {}, ""))
            pending += 1
            continue
        d = r.get("device") or {}
        prec = r.get("cell", {}).get("precision", "")
        dev = f"{d.get('chip', '?')} {d.get('memory_gb', 0)}GB"
        if prec:
            dev += f" · {prec}"
        if not r.get("ok"):
            reason = (r.get("errors") or ["unknown error"])[0]
            # A device that declined because it was busy is not a crash and not
            # a regression: the cell never measured, so it is environmental and
            # re-runnable. Show it apart from a real failure and keep it out of
            # the failure count so a busy fleet does not read as a broken PR.
            if r.get("declined"):
                rows.append(_row(c["id"], "busy", dev, {}, reason[:48]))
                declined += 1
            else:
                rows.append(_row(c["id"], "failed", dev, {}, reason[:48]))
                failed += 1
            continue
        if not r["delta"] and any(
            k in r.get("cell", {}) for k in ("greedy_agreement",)
        ):
            rows.append(_parity_row(c["id"], dev, r["cell"]))
            ok += 1
            continue
        state = _cell_state(r["delta"])
        rows.append(
            _row(
                c["id"],
                state,
                dev,
                r["delta"],
                "re-run on an idle device" if state == "inconclusive" else "",
            )
        )
        ok += 1

    # The summary table, for a single-architecture change, is the headline and
    # always drawn. Its regression count drives the status so the two agree.
    if one_arch:
        _, regressed = _summary(cells, results)

    done = ok + failed + declined
    if pending:
        head = f"running — {done} of {len(cells)} done"
    elif regressed:
        head = f"{regressed} regression(s)"
    elif failed:
        # A crash is not hidden behind a passing headline. Surfacing it here
        # also drives the commit status red, so a maintainer sees it on the PR.
        head = "all cells failed" if not ok else f"{failed} cell(s) failed"
    elif ok == 0 and declined:
        head = "all devices busy — re-run when idle"
    else:
        head = "no regression"

    # Barebone by request: a title and one table, nothing else. The title
    # carries the verdict; the status column carries each cell's outcome.
    title = f"`{cells[0]['arch']}` — {head}" if one_arch else head
    lines = [marker(pr), f"### mlx-vlm benchmark · {title}", ""]
    if warning:
        lines += [f"> ⚠️ {warning}", ""]
    lines += [
        "| cell | device | status | "
        + " | ".join(METRIC_LABEL.get(m, m) for m in SPEED)
        + " | note |",
        "|" + "---|" * (len(SPEED) + 4),
    ]
    lines += rows
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


def set_status(
    repo: str, sha: str, state: str, description: str, url: str = ""
) -> None:
    """Post a commit status so the result shows on the PR, not only in a comment.

    An issue_comment run is not linked to the pull request as a check, so
    without this the PR's checks area stays empty even while the benchmark runs
    -- which reads as "CI never ran".
    """
    _api(
        "POST",
        f"/repos/{repo}/statuses/{sha}",
        {
            "state": state,
            "context": "benchmark",
            "description": description[:140],
            **({"target_url": url} if url else {}),
        },
    )


def status_for(comment: str) -> tuple:
    """Map a rendered comment to (commit-status state, description). Keyed on the
    one headline the report already computes, so the status and the comment can
    never disagree."""
    head = next(
        (ln for ln in comment.splitlines() if "mlx-vlm benchmark" in ln),
        "",
    )
    low = head.lower()
    if "running" in low:
        state = "pending"
    elif "regression(s)" in low or "failed" in low or "refused" in low:
        state = "failure"
    else:
        state = "success"
    desc = head.split("benchmark", 1)[-1].lstrip(" ·—#*").strip() or "benchmark"
    return state, desc


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
    ap.add_argument("--head", help="PR head sha, to post a commit status")
    ap.add_argument("--results", help="directory of result json; omit for pending")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells).read_text())
    # Notes carry the "nothing to run" reason and the CI-change warning/refusal;
    # they were parsed but never handed to render, so they never reached the
    # comment.
    notes = json.loads(Path(args.notes).read_text()) if args.notes else None
    results = None
    if args.results:
        results = [
            json.loads(p.read_text()) for p in Path(args.results).rglob("result.json")
        ]
    comment = render(args.pr, cells, results, notes=notes)
    upsert(args.pr, args.repo, comment)
    if args.head:
        state, desc = status_for(comment)
        set_status(args.repo, args.head, state, desc, os.environ.get("CI_RUN_URL", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
