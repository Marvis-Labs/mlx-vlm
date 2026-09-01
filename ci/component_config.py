from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from ci.components.registry import contributor_config_paths


def materialize(
    repository: Path,
    revision: str,
    output: Path,
    paths: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    selected = tuple(paths) if paths is not None else contributor_config_paths()
    written: list[Path] = []
    for relative_path in selected:
        result = subprocess.run(
            ["git", "show", f"{revision}:ci/{relative_path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.stdout)
        written.append(destination)
    return tuple(written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    materialize(args.repository, args.revision, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
