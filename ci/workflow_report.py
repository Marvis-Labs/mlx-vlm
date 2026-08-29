from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.bot import BotOutput
from ci.scheduler import bot_result


def report(
    plan: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    run_url: str,
) -> dict[str, Any]:
    if result is not None:
        execution = dict(result)
    elif dispatch.get("outcome") == "no_eligible_runner":
        execution = bot_result(dispatch)
    else:
        job = dispatch.get("job", {})
        execution = {
            "component": str(job.get("component", "runner")),
            "model": job.get("model"),
            "mode": str(job.get("mode", "default")),
            "job_id": str(job.get("id", "")),
            "outcome": "infrastructure_failure",
        }
    record = dict(plan)
    record.update(
        {
            "kind": "ci_execution",
            "outcome": execution["outcome"],
            "run_url": run_url,
            "results": [execution],
        }
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    dispatch = json.loads(args.dispatch.read_text())
    result = json.loads(args.result.read_text()) if args.result.is_file() else None
    record = report(plan, dispatch, result, args.run_url)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(BotOutput(record).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
