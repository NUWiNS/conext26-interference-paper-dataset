#!/usr/bin/env python3
"""Build the 2022 intermediate data from the released raw XCAL and LTE logs."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = ("clean_2022.py", "parse_2022_lte_neighbors.py", "build_coverage_2022.py")
BAND_SUMMARY = ROOT / "data/intermediate_data/2024/all_cells_band_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List steps without running them")
    args = parser.parse_args()

    if not args.list and not BAND_SUMMARY.is_file():
        parser.error(
            f"2022 cleaning needs {BAND_SUMMARY}. "
            "Run run_all_2024.py first to build the band summary."
        )

    for index, name in enumerate(STEPS, 1):
        script = SCRIPT_DIR / name
        print(f"[{index}/{len(STEPS)}] {name}", flush=True)
        if args.list:
            continue
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"{name} failed with exit code {result.returncode}")

    if not args.list:
        print("Finished all 2022 data cleaning steps.")


if __name__ == "__main__":
    main()
