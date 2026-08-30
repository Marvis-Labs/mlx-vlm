"""Where measurements accumulate. SQLite, one file, no server.

Results are observational: they pile up forever, are queried across revisions
and devices, and cannot be committed to git the way declarative data is. This
is deliberately the only stateful part of the CI -- everything else is a file
in the repo -- and it stays a single SQLite file so there is nothing to run.

Two things read from it. The router, to size a cell from the memory a prior
run actually measured rather than an estimate. A person, to ask when a metric
regressed over time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path.home() / ".local" / "state" / "marvis-ci" / "results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id         INTEGER PRIMARY KEY,
    cell       TEXT NOT NULL,     -- stable cell id, no precision
    arch       TEXT NOT NULL,
    component  TEXT NOT NULL,
    precision  TEXT,              -- what the device chose
    device     TEXT,              -- chip + memory
    head_sha   TEXT,
    ok         INTEGER,
    peak_mem_gb REAL,             -- measured, for sizing feedback
    delta      TEXT,              -- the full comparison, as json
    measured_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS results_cell ON results(cell, measured_at);
CREATE INDEX IF NOT EXISTS results_arch ON results(arch, component);
"""


def connect(db: Optional[Path] = None) -> sqlite3.Connection:
    db = Path(db or DEFAULT_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(_SCHEMA)
    return con


def record(result: dict, head_sha: str = "", db: Optional[Path] = None) -> None:
    """Store one cell result. Idempotent enough: history is append-only."""
    cell = result.get("cell", {})
    dev = result.get("device", {})
    head = cell.get("peak_mem_gb")
    if head is None:  # pull the measured peak from the head summary if present
        head = (result.get("head") or {}).get("peak_mem_gb", {})
        head = head.get("median") if isinstance(head, dict) else None
    con = connect(db)
    with con:
        con.execute(
            "INSERT INTO results (cell, arch, component, precision, device, "
            "head_sha, ok, peak_mem_gb, delta) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                cell.get("id", "?"),  # the id is already precision-free
                cell.get("arch", "?"),
                cell.get("component", "?"),
                cell.get("precision"),
                f"{dev.get('chip','?')} {dev.get('memory_gb',0)}GB",
                head_sha,
                int(bool(result.get("ok"))),
                head,
                json.dumps(result.get("delta", {})),
            ),
        )
    con.close()


def measured_peak(
    arch: str, precision: str, db: Optional[Path] = None
) -> Optional[float]:
    """The most recent measured peak memory for an (arch, precision), if any.

    This is what replaces the weights-times-multiplier estimate: a size the
    hardware actually reported beats a guess.
    """
    con = connect(db)
    row = con.execute(
        "SELECT peak_mem_gb FROM results WHERE arch=? AND precision=? "
        "AND peak_mem_gb IS NOT NULL ORDER BY measured_at DESC LIMIT 1",
        (arch, precision),
    ).fetchone()
    con.close()
    return row[0] if row else None


def history(
    cell: str, limit: int = 30, db: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Recent measurements for one cell, newest first -- for trend queries."""
    con = connect(db)
    rows = con.execute(
        "SELECT measured_at, device, precision, ok, delta FROM results "
        "WHERE cell=? ORDER BY measured_at DESC LIMIT ?",
        (cell, limit),
    ).fetchall()
    con.close()
    return [
        {
            "at": r[0],
            "device": r[1],
            "precision": r[2],
            "ok": bool(r[3]),
            "delta": json.loads(r[4] or "{}"),
        }
        for r in rows
    ]


def ingest(results_dir, head_sha: str = "", db: Optional[Path] = None) -> int:
    """Record every result.json under a directory. Returns the count.

    Called by the report job after a run, so the measurements a run produced
    accumulate in the store on the results-data branch rather than vanishing
    with the ephemeral cloud job.
    """
    n = 0
    for path in Path(results_dir).rglob("result.json"):
        try:
            record(json.loads(path.read_text()), head_sha, db)
            n += 1
        except Exception:
            continue
    return n


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", help="directory of result.json to record")
    ap.add_argument("--head", default="")
    ap.add_argument("--db")
    args = ap.parse_args()
    if args.ingest:
        db = Path(args.db) if args.db else None
        print(f"recorded {ingest(args.ingest, args.head, db)} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
