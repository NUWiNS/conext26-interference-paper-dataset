#!/usr/bin/env python3
"""
Build the 2024 LTE neighbor data from the all-cell report, Qualcomm TXT logs,
and route area annotations. This single command writes:
  - data/intermediate_data/2024/frequency_label_lookup.csv
  - data/intermediate_data/2024/neighbor_lte_pcell.csv
  - data/intermediate_data/2024/lte_channel_neighbor_summary.csv

Each .txt file contains repeated [0xB179] "LTE ML1 Connected mode LTE intrafrequency
measurement results" blocks. For each block where Serving_Cell_Idx == "PCell", emit
one output row carrying:
  - source_file, operator (A->att, T->tmobile, V->verizon)
  - time (from the block title)
  - earfcn, frequency_mhz, pci, rsrp_dbm, rsrq_db
  - neighbor_count (Number_of_measured_neighbor_cells_N -- 0 is preserved)
  - frequency_label, band, band_type (looked up from the all-cell report)
  - area (latest preceding local-time annotation from data/raw_data/2024/area.csv)
"""
from __future__ import annotations

import argparse
import csv
import re
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

OP_LETTER_MAP = {"A": "att", "T": "tmobile", "V": "verizon"}


def build_frequency_label_lookup(cell_report: Path, output: Path) -> None:
    """Reduce the all-cell report to one label per operator, RAT, and channel."""
    df = pd.read_csv(
        cell_report,
        usecols=["operator", "rat", "channel_key", "band", "frequency_label", "band_type"],
    )
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].astype(str).str.upper().replace({"5G": "NR", "4G": "LTE"})
    grouped = (
        df.groupby(["operator", "rat", "channel_key"])
          .agg(
              band=("band", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
              frequency_label=(
                  "frequency_label",
                  lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
              ),
              band_type=(
                  "band_type",
                  lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0],
              ),
              n_cells=("frequency_label", "size"),
          )
          .reset_index()
          .sort_values(["operator", "rat", "band", "channel_key"])
          .reset_index(drop=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output, index=False)
    print(f"[OK] wrote {output} ({len(grouped)} rows)")

# Title line. Two formats observed in 2024 dump:
#   5-col: <idx>\t<time>\t<Code e.g. "Qc. LTE[Chipset]">\t<Message>\t<Detail>
#   4-col: <idx>\t<time>\t<Message>\t<Detail>
# So the Code column is optional.
TITLE_RE = re.compile(
    r"^(?P<idx>\d+)\t"
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\t"
    r"(?:[^\t]*\t)?"  # optional Code column (5-col format)
    r"______\[0xB179\] LTE ML1 Connected mode LTE intrafrequency measurement results"
)

EARFCN_RE = re.compile(r"^E_ARFCN\s*:\s*(\d+)\((\d+(?:\.\d+)?)MHz\)")
SCI_RE = re.compile(r"^Serving_Cell_Idx\s*:\s*(\S+)")
PCI_RE = re.compile(r"^Serving_physical_cell_ID\s*:\s*(\d+)")
RSRP_RE = re.compile(r"^Serving_filtered_RSRP\s*:\s*(-?\d+(?:\.\d+)?)\s*dBm")
RSRQ_RE = re.compile(r"^Serving_filtered_RSRQ\s*:\s*(-?\d+(?:\.\d+)?)\s*dB")
NEI_RE = re.compile(r"^Number_of_measured_neighbor_cells_N\s*:\s*(\d+)")


def load_area_annotations(path: Path) -> tuple[list[datetime], list[str]]:
    """Use local wall time because the Qualcomm log title has no UTC offset."""
    annotations: list[tuple[datetime, str]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = (row.get("value") or "").strip()
            if not value:
                continue
            try:
                local_time = datetime.fromisoformat(row["local_dt"]).replace(tzinfo=None)
            except (TypeError, ValueError):
                continue
            annotations.append((local_time, value))
    if not annotations:
        raise ValueError(f"No usable area annotations in {path}")
    annotations.sort(key=lambda item: item[0])
    return [item[0] for item in annotations], [item[1] for item in annotations]


def emit(writer: csv.writer, source_file: str, op: str, blk: dict,
         lookup: Dict[Tuple[str, str, str], Tuple[str, str, str]],
         area_times: list[datetime], area_labels: list[str]) -> None:
    if blk.get("cell_idx") != "PCell":
        return
    if blk.get("earfcn") is None:
        return
    if blk.get("nei") is None:
        return
    channel_key = blk["earfcn"]
    band, frequency_label, band_type = lookup.get(
        (op, "LTE", str(channel_key)), ("", "", "")
    )
    area_index = bisect_right(area_times, datetime.fromisoformat(blk["time"])) - 1
    area = area_labels[max(area_index, 0)]
    writer.writerow([
        source_file, op, blk.get("time", ""),
        channel_key, blk.get("freq_mhz", ""),
        blk.get("pci", ""),
        blk.get("rsrp", ""), blk.get("rsrq", ""),
        blk["nei"],
        band, frequency_label, band_type, area,
    ])


def parse_one_file(path: Path, op: str,
                    lookup: Dict[Tuple[str, str, str], Tuple[str, str, str]],
                    writer: csv.writer, area_times: list[datetime],
                    area_labels: list[str]) -> Tuple[int, int]:
    """Returns (n_blocks_total, n_pcell_emitted)."""
    n_total = 0
    n_emit = 0
    blk: dict = {}
    with path.open() as f:
        f.readline()  # header
        for line in f:
            m = TITLE_RE.match(line)
            if m:
                # Flush previous block
                if blk:
                    n_total += 1
                    if blk.get("cell_idx") == "PCell" and blk.get("earfcn") is not None and blk.get("nei") is not None:
                        emit(writer, path.name, op, blk, lookup, area_times, area_labels)
                        n_emit += 1
                blk = {"time": m.group("time")}
                continue
            # Key/value lines
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
    # flush last
    if blk:
        n_total += 1
        if blk.get("cell_idx") == "PCell" and blk.get("earfcn") is not None and blk.get("nei") is not None:
            emit(writer, path.name, op, blk, lookup, area_times, area_labels)
            n_emit += 1
    return n_total, n_emit


def load_lookup(path: Path) -> Dict[Tuple[str, str, str], Tuple[str, str, str]]:
    df = pd.read_csv(path)
    out: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}
    for r in df.itertuples(index=False):
        key = (str(r.operator).lower(), str(r.rat).upper(), str(int(float(r.channel_key))))
        out[key] = (str(r.band), str(r.frequency_label), str(r.band_type))
    return out


def parse_neighbor_logs(in_dir: Path, lookup_csv: Path, area_csv: Path, output: Path) -> None:
    lookup = load_lookup(lookup_csv)
    area_times, area_labels = load_area_annotations(area_csv)
    print(f"[INFO] lookup entries: {len(lookup)}")
    print(f"[INFO] area annotations: {len(area_times)}")

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No LTE neighbor TXT logs found in {in_dir}")
    print(f"[INFO] parsing {len(files)} files")

    output.parent.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    grand_emit = 0
    with output.open("w", newline="") as outf:
        w = csv.writer(outf)
        w.writerow([
            "source_file", "operator", "time",
            "channel_key", "frequency_mhz",
            "pci", "rsrp_dbm", "rsrq_db",
            "neighbor_count",
            "band", "frequency_label", "band_type", "area",
        ])
        for path in files:
            letter = path.name[0].upper()
            op = OP_LETTER_MAP.get(letter)
            if op is None:
                print(f"[WARN] cannot infer operator from {path.name}; skipping")
                continue
            n_total, n_emit = parse_one_file(path, op, lookup, w, area_times, area_labels)
            grand_total += n_total
            grand_emit += n_emit
            print(f"  {path.name}  blocks={n_total:>7,}  pcell_emitted={n_emit:>7,}")

    print(f"\n[OK] wrote {output}")
    print(f"     total blocks parsed: {grand_total:,}")
    print(f"     PCell rows emitted:  {grand_emit:,}")


def build_channel_neighbor_summary(neighbor_csv: Path, output: Path) -> None:
    """Summarize LTE PCell neighbors by operator and channel for Fig. 7."""
    df = pd.read_csv(neighbor_csv)
    df["operator"] = df["operator"].astype(str).str.strip().str.lower()
    df["channel_key"] = df["channel_key"].astype(str).str.strip()
    df["pci"] = df["pci"].astype(str).str.strip()
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")

    keys = ["operator", "channel_key"]
    grouped = df.groupby(keys, dropna=False)
    summary = grouped.agg(
        num_cells=("pci", "nunique"),
        avg_neighbor_count=("neighbor_count", "mean"),
        n_records=("neighbor_count", "size"),
    ).reset_index()
    carry = ["band", "frequency_label", "band_type"]
    summary = summary.merge(grouped[carry].first().reset_index(), on=keys, how="left")
    summary = summary[
        [
            "operator", "channel_key", "band", "frequency_label", "band_type",
            "num_cells", "avg_neighbor_count", "n_records",
        ]
    ].sort_values(["operator", "channel_key"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"[OK] wrote {output} ({len(summary)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse 2024 LTE neighbor logs and generate all three intermediate CSVs."
    )
    ap.add_argument(
        "--cell-report", type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/all_cells_aggregated_report.csv",
    )
    ap.add_argument(
        "--in-dir", type=Path,
        default=REPO_ROOT / "data/raw_data/2024/lte_neighbor",
    )
    ap.add_argument(
        "--area-csv", type=Path,
        default=REPO_ROOT / "data/raw_data/2024/area.csv",
        help="Local-time route area annotations",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024",
    )
    args = ap.parse_args()

    lookup_csv = args.out_dir / "frequency_label_lookup.csv"
    neighbor_csv = args.out_dir / "neighbor_lte_pcell.csv"
    summary_csv = args.out_dir / "lte_channel_neighbor_summary.csv"
    build_frequency_label_lookup(args.cell_report, lookup_csv)
    parse_neighbor_logs(args.in_dir, lookup_csv, args.area_csv, neighbor_csv)
    build_channel_neighbor_summary(neighbor_csv, summary_csv)


if __name__ == "__main__":
    main()
