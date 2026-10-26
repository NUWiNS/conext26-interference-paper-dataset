#!/usr/bin/env python3
"""Regenerate all paper figures from the data in this repository."""

import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PLOTTING_DIR = Path(__file__).resolve().parent


def figure_number(path: Path) -> tuple[int, str]:
    match = re.match(r"fig_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected plotting script name: {path.name}")
    return int(match.group(1)), path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List scripts without running them")
    args = parser.parse_args()

    scripts = sorted(PLOTTING_DIR.glob("fig_*.py"), key=figure_number)
    if not scripts:
        raise SystemExit(f"No figure scripts found in {PLOTTING_DIR}")

    for index, script in enumerate(scripts, 1):
        print(f"[{index}/{len(scripts)}] {script.name}", flush=True)
        if args.list:
            continue
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"{script.name} failed with exit code {result.returncode}")

    if not args.list:
        print(f"Finished all {len(scripts)} figure scripts.")


if __name__ == "__main__":
    main()
