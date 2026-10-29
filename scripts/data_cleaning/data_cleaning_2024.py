#!/usr/bin/env python3
"""Build the 2024 intermediate data from the released inputs and raw logs."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts"
STEPS = (
    ("data_cleaning/build_all_cell_reports.py", ()),
    ("data_cleaning/clean_2024.py", ()),
    ("data_cleaning/build_complete_ue_level_log.py", ()),
    ("data_cleaning/parse_2024_lte_neighbors.py", ()),
    ("data_cleaning/build_coverage_2024.py", ()),
    ("plotting/fig_11_12_13_16.py", ("--rebuild", "--correlations-only")),
)
CELL_INPUT_DIR = ROOT / "data/intermediate_data/2024/cell_report_inputs"
APP_ROOT = ROOT / "data/raw_data/2024/app_data_all"
CELL_INPUTS = (
    "fig3_cell_perspective_report.csv",
    "fig45_ue_record_band_type_summary.csv",
    "area_type_per_cell.csv",
    "timeofday_boxplot_points.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List steps without running them")
    args = parser.parse_args()

    if not args.list:
        missing = [name for name in CELL_INPUTS if not (CELL_INPUT_DIR / name).is_file()]
        if missing:
            parser.error(
                f"build_all_cell_reports.py needs archived cell_report_inputs in {CELL_INPUT_DIR}: "
                + ", ".join(missing)
            )
        if not any(APP_ROOT.glob("*/*/*/tcp_downlink*xcal.csv")):
            parser.error(f"No raw application downlink CSVs found under {APP_ROOT}")

    for index, (relative_path, extra_args) in enumerate(STEPS, 1):
        script = SCRIPT_ROOT / relative_path
        print(f"[{index}/{len(STEPS)}] {relative_path} {' '.join(extra_args)}".rstrip(), flush=True)
        if args.list:
            continue
        result = subprocess.run([sys.executable, str(script), *extra_args], cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"{relative_path} failed with exit code {result.returncode}")

    if not args.list:
        print("Finished all 2024 data cleaning steps.")


if __name__ == "__main__":
    main()
