#!/usr/bin/env python3
"""
Parse 2022 Qualcomm LTE intrafreq neighbor logs from data/raw_data/2022/lte_neighbor/.

Each .txt holds repeated [0xB179] "LTE ML1 Connected mode LTE intrafrequency
measurement results" blocks. For each PCell block we record the chipset event
time (the bracket line, which matches the TIME_STAMP format in
operator_runs_log.csv -- the title-line Time column is the analyzer wall clock
and is unreliable in this dataset).

For each block we:
  - parse the bracket time and key/value fields
  - look up the operator and run_idx via operator_runs_log.csv
      key: source_file_stem.csv + bracket_time in [start_TIME_STAMP, end_TIME_STAMP]
  - look up frequency_label / band / band_type via per_cell_all.csv (LTE rows,
    keyed by (operator, channel_key=earfcn))

Writes: data/intermediate_data/2022/neighbor_lte_pcell.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MONTH = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}

# operator_runs_log uses display names; per_cell_all.csv uses lowercase shortnames.
# Normalize for the (operator, earfcn) -> frequency_label lookup.
OPERATOR_TO_SHORT = {
    "AT&T": "att",
    "T-Mobile": "tmobile",
    "Verizon": "verizon",
    "att": "att",
    "tmobile": "tmobile",
    "verizon": "verizon",
}

# Title line. Handles both 4-col (no Code column) and 5-col (with Code) formats.
TITLE_RE = re.compile(
    r"^(?P<idx>\d+)\t"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\t"  # analyzer wall-clock; ignored
    r"(?:[^\t]*\t)?"
    r"______\[0xB179\] LTE ML1 Connected mode LTE intrafrequency measurement results"
)
# Bracket: [ YYYY  Mon  D    HH:MM:SS.ffffff ]  -- the chipset event time
BRACKET_RE = re.compile(
    r"^\[\s*(?P<y>\d{4})\s+(?P<mon>\w{3})\s+(?P<d>\d+)\s+(?P<t>\d{2}:\d{2}:\d{2}\.\d+)\s*\]"
)
EARFCN_RE = re.compile(r"^E_ARFCN\s*:\s*(\d+)\((\d+(?:\.\d+)?)MHz\)")
SCI_RE = re.compile(r"^Serving_Cell_Idx\s*:\s*(\S+)")
PCI_RE = re.compile(r"^Serving_physical_cell_ID\s*:\s*(\d+)")
RSRP_RE = re.compile(r"^Serving_filtered_RSRP\s*:\s*(-?\d+(?:\.\d+)?)\s*dBm")
RSRQ_RE = re.compile(r"^Serving_filtered_RSRQ\s*:\s*(-?\d+(?:\.\d+)?)\s*dB")
NEI_RE = re.compile(r"^Number_of_measured_neighbor_cells_N\s*:\s*(\d+)")


def parse_bracket_time(line: str) -> Optional[str]:
    """Return 'YYYY-MM-DD HH:MM:SS.ffffff' or None."""
    m = BRACKET_RE.match(line)
    if not m:
        return None
    mon = MONTH.get(m.group("mon"))
    if mon is None:
        return None
    return f"{m.group('y')}-{mon:02d}-{int(m.group('d')):02d} {m.group('t')}"


# ---------------------------------------------------------------------------
# Operator-runs lookup
# ---------------------------------------------------------------------------
def load_run_index(path: Path) -> Dict[str, List[Tuple[str, str, str, int]]]:
    """
    source_file_stem (no .csv) -> list of (start_TIME_STAMP, end_TIME_STAMP,
                                            operator, run_idx)  sorted by start.
    Uses string sort on the TIME_STAMP column (lexicographic == chronological
    for 'YYYY-MM-DD HH:MM:SS.ffffff').
    """
    df = pd.read_csv(path)
    out: Dict[str, List[Tuple[str, str, str, int]]] = {}
    for r in df.itertuples(index=False):
        if pd.isna(r.start_TIME_STAMP) or pd.isna(r.end_TIME_STAMP):
            continue
        stem = Path(str(r.source_file)).stem
        out.setdefault(stem, []).append(
            (str(r.start_TIME_STAMP), str(r.end_TIME_STAMP), str(r.operator), int(r.run_idx))
        )
    for stem in out:
        out[stem].sort(key=lambda x: x[0])
    return out


def find_run(runs: List[Tuple[str, str, str, int]], t: str) -> Optional[Tuple[str, int]]:
    """Binary-search the runs by start_TIME_STAMP; verify end >= t. Returns (op, run_idx)."""
    if not runs:
        return None
    starts = [r[0] for r in runs]
    i = bisect_right(starts, t) - 1
    if i < 0:
        return None
    start, end, op, idx = runs[i]
    if t > end:
        return None
    return op, idx


# ---------------------------------------------------------------------------
# frequency_label lookup (per operator, LTE channel_key=earfcn)
# ---------------------------------------------------------------------------
def load_freq_label_lookup(per_cell_csv: Path) -> Dict[Tuple[str, str], Tuple[str, str, str]]:
    """(operator_lower, earfcn_str) -> (band, frequency_label, band_type)"""
    df = pd.read_csv(
        per_cell_csv, low_memory=False,
        usecols=["operator", "rat", "channel_key", "band", "frequency_label", "band_type"],
    )
    df = df[df["rat"].astype(str).str.upper().isin(["LTE", "4G"])]
    df = df.dropna(subset=["operator", "channel_key"]).copy()
    df["operator"] = df["operator"].astype(str).str.lower()
    df["channel_key"] = df["channel_key"].astype(str).str.replace(r"\.0$", "", regex=True)

    grp = (
        df.groupby(["operator", "channel_key"])
          .agg(
              band=("band", lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else ""),
              frequency_label=(
                  "frequency_label",
                  lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "",
              ),
              band_type=(
                  "band_type",
                  lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "",
              ),
          )
          .reset_index()
    )
    return {
        (r.operator, r.channel_key): (r.band, r.frequency_label, r.band_type)
        for r in grp.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_one_file(
    path: Path,
    run_index: Dict[str, List[Tuple[str, str, str, int]]],
    label_lookup: Dict[Tuple[str, str], Tuple[str, str, str]],
    writer: csv.writer,
) -> Tuple[int, int, int]:
    """Returns (n_blocks_total, n_pcell_emitted, n_pcell_no_op_match)."""
    stem = path.stem
    runs = run_index.get(stem, [])
    n_total = n_emit = n_no_op = 0
    blk: dict = {}

    def flush(b: dict) -> None:
        nonlocal n_total, n_emit, n_no_op
        if not b:
            return
        n_total += 1
        if b.get("cell_idx") != "PCell":
            return
        if b.get("earfcn") is None or b.get("nei") is None or b.get("time") is None:
            return
        match = find_run(runs, b["time"])
        if match is None:
            n_no_op += 1
            op, run_idx = "unknown", -1
        else:
            op, run_idx = match
        ck = b["earfcn"]
        op_short = OPERATOR_TO_SHORT.get(op, op.lower())
        band, freq_label, band_type = label_lookup.get(
            (op_short, ck), ("", "", "")
        )
        writer.writerow([
            path.name, op, run_idx, b["time"],
            ck, b.get("freq_mhz", ""),
            b.get("pci", ""),
            b.get("rsrp", ""), b.get("rsrq", ""),
            b["nei"],
            band, freq_label, band_type,
        ])
        n_emit += 1

    with path.open() as f:
        f.readline()  # header
        for line in f:
            if TITLE_RE.match(line):
                # Title means start of a new block. Don't flush yet -- the bracket
                # line that follows carries the actual chipset event time.
                flush(blk)
                blk = {}
                continue
            m = BRACKET_RE.match(line)
            if m:
                bt = parse_bracket_time(line)
                if bt is not None:
                    blk["time"] = bt
                continue
            mm = EARFCN_RE.match(line)
            if mm:
                blk["earfcn"] = mm.group(1)
                blk["freq_mhz"] = mm.group(2)
                continue
            mm = SCI_RE.match(line)
            if mm:
                blk["cell_idx"] = mm.group(1)
                continue
            mm = PCI_RE.match(line)
            if mm:
                blk["pci"] = mm.group(1)
                continue
            mm = RSRP_RE.match(line)
            if mm:
                blk["rsrp"] = mm.group(1)
                continue
            mm = RSRQ_RE.match(line)
            if mm:
                blk["rsrq"] = mm.group(1)
                continue
            mm = NEI_RE.match(line)
            if mm:
                blk["nei"] = int(mm.group(1))
                continue
    flush(blk)
    return n_total, n_emit, n_no_op


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=REPO_ROOT / "data" / "raw_data" / "2022" / "lte_neighbor")
    ap.add_argument(
        "--runs-log",
        type=Path,
        default=REPO_ROOT / "data" / "intermediate_data" / "2022" / "operator_runs_log.csv",
    )
    ap.add_argument(
        "--per-cell-csv",
        type=Path,
        default=REPO_ROOT / "data" / "intermediate_data" / "2022" / "per_cell_all.csv",
        help="Used to build the (operator, earfcn) -> frequency_label lookup",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "intermediate_data" / "2022" / "neighbor_lte_pcell.csv",
    )
    args = ap.parse_args()

    run_index = load_run_index(args.runs_log)
    print(f"[INFO] operator_runs_log: {sum(len(v) for v in run_index.values()):,} runs "
          f"across {len(run_index)} source files")

    label_lookup = load_freq_label_lookup(args.per_cell_csv)
    print(f"[INFO] frequency_label lookup: {len(label_lookup)} (operator, earfcn) entries")

    files = sorted(args.in_dir.glob("*.txt"))
    if not files:
        ap.error(f"No LTE neighbor logs found in {args.in_dir}")
    print(f"[INFO] parsing {len(files)} files")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grand_total = grand_emit = grand_no_op = 0
    with args.output.open("w", newline="") as outf:
        w = csv.writer(outf)
        w.writerow([
            "source_file", "operator", "run_idx", "time",
            "channel_key", "frequency_mhz",
            "pci", "rsrp_dbm", "rsrq_db",
            "neighbor_count",
            "band", "frequency_label", "band_type",
        ])
        for path in files:
            n_total, n_emit, n_no_op = parse_one_file(path, run_index, label_lookup, w)
            grand_total += n_total
            grand_emit += n_emit
            grand_no_op += n_no_op
            print(
                f"  {path.name:65s}  blocks={n_total:>7,}  "
                f"pcell={n_emit:>7,}  no_op_match={n_no_op:>4,}"
            )

    print(f"\n[OK] wrote {args.output}")
    print(f"     total blocks parsed: {grand_total:,}")
    print(f"     PCell rows emitted:  {grand_emit:,}  "
          f"(of which {grand_no_op:,} had no operator run match)")


if __name__ == "__main__":
    main()
