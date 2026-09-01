from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--synthetic-compare", type=Path)
    parser.add_argument("--synthetic-probe", type=Path)
    parser.add_argument("--hf-compare", type=Path)
    parser.add_argument("--hf-probe", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args(argv)

    control = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(control))
    work_executor = importlib.import_module("ci.work_executor")
    translated = [
        "--job",
        str(args.job),
        "--control",
        str(control),
        "--base",
        str(args.base),
        "--head",
        str(args.head),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.image is not None:
        translated.extend(("--image", str(args.image)))
    return work_executor.main(translated)


if __name__ == "__main__":
    raise SystemExit(main())
