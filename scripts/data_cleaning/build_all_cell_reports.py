#!/usr/bin/env python3
"""
Build the all-cell reports by joining the archived cell-level inputs.

Base:
  data/intermediate_data/2024/cell_report_inputs/fig3_cell_perspective_report.csv

Added columns:
  - channel_key extracted from pcell_key
  - frequency_label and band_type from the UE-record band-type summary
  - area_type from cell_report_inputs/area_type_per_cell.csv
  - period from cell_report_inputs/timeofday_boxplot_points.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data/intermediate_data/2024/cell_report_inputs"
DEFAULT_BASE_CSV = INPUT_DIR / "fig3_cell_perspective_report.csv"
DEFAULT_UE_BAND_CSV = (
    INPUT_DIR / "fig45_ue_record_band_type_summary.csv"
)
DEFAULT_AREA_CSV = INPUT_DIR / "area_type_per_cell.csv"
DEFAULT_TIME_CSV = INPUT_DIR / "timeofday_boxplot_points.csv"
DEFAULT_OUT_CSV = ROOT / "data/intermediate_data/2024/all_cells_aggregated_report.csv"
DEFAULT_BAND_SUMMARY_CSV = ROOT / "data/intermediate_data/2024/all_cells_band_summary.csv"
DEFAULT_BAND_CHANNEL_SUMMARY_CSV = ROOT / "data/intermediate_data/2024/all_cells_band_channel_summary.csv"

BAND_TYPE_ORDER = {"low": 0, "mid": 1, "mmwave": 2, "unknown": 3}

LTE_DL_BAND_TABLE = {
    # band: (F_DL_low_MHz, NOffs_DL, EARFCN_min, EARFCN_max)
    2: (1930.0, 600, 600, 1199),
    4: (2110.0, 1950, 1950, 2399),
    5: (869.0, 2400, 2400, 2649),
    12: (729.0, 5010, 5010, 5179),
    13: (746.0, 5180, 5180, 5279),
    14: (758.0, 5280, 5280, 5379),
    17: (734.0, 5730, 5730, 5849),
    25: (1930.0, 8040, 8040, 8689),
    26: (859.0, 8690, 8690, 9039),
    29: (717.0, 9660, 9660, 9769),
    30: (2350.0, 9770, 9770, 9869),
    41: (2496.0, 39650, 39650, 41589),
    66: (2110.0, 66436, 66436, 67335),
    71: (617.0, 68586, 68586, 68935),
}


def clean_value(x) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]
    return s


def normalize_operator(x) -> str:
    return str(x).strip().lower()


def normalize_rat(x) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"NR", "5G"}:
        return "5G"
    if s in {"LTE", "4G"}:
        return "4G"
    return None


def extract_channel_key(pcell_key: str, rat: Optional[str]) -> Optional[str]:
    key = str(pcell_key)
    if rat == "5G":
        field = "freq"
    elif rat == "4G":
        field = "earfcn"
    else:
        field = "freq" if key.startswith("NR|") else "earfcn"
    m = re.search(rf"(?:^|\|){field}=([^|]+)", key)
    return clean_value(m.group(1)) if m else None


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def parse_float(x) -> Optional[float]:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def extract_band_number(band) -> Optional[int]:
    s = clean_value(band)
    if s is None:
        return None
    m = re.search(r"\bBand\s*(\d+)\b", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bb(\d+)\b", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bn(\d+)\b", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def lte_dl_freq_mhz(band, earfcn) -> Optional[float]:
    band_num = extract_band_number(band)
    earfcn_num = parse_float(earfcn)
    if band_num is None or earfcn_num is None:
        return None

    earfcn_int = int(round(earfcn_num))
    if band_num not in LTE_DL_BAND_TABLE:
        return None

    f_dl_low, n_offs_dl, n_min, n_max = LTE_DL_BAND_TABLE[band_num]
    if not (n_min <= earfcn_int <= n_max):
        return None
    return f_dl_low + 0.1 * (earfcn_int - n_offs_dl)


def band_type_from_freq_mhz(freq_mhz: Optional[float], rat: str) -> str:
    if freq_mhz is None or not math.isfinite(freq_mhz):
        return "unknown"
    if freq_mhz < 1000:
        return "low"
    if rat == "5G" and freq_mhz >= 24000:
        return "mmwave"
    return "mid"


def normalize_existing_band_type(x) -> Optional[str]:
    s = clean_value(x)
    if s is None:
        return None
    s = s.lower()
    if s in {"low", "mid", "mmwave"}:
        return s
    if s in {"mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return None


def join_unique(values: Iterable) -> str:
    cleaned = sorted({str(v) for v in values if clean_value(v) is not None})
    return ",".join(cleaned)


def build_band_channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        DEFAULT_OUT_CSV,
        {"operator", "rat", "pcell_key", "band", "channel_key"},
    )
    work = df.copy()
    work["operator"] = work["operator"].map(normalize_operator)
    work["rat_norm"] = work["rat"].map(normalize_rat)
    work["band"] = work["band"].map(clean_value)
    work["channel_key"] = work["channel_key"].map(clean_value)
    work["channel_value"] = work["channel_key"].map(parse_float)

    if "band_type" not in work.columns:
        work["band_type"] = None

    def classify_row(row: pd.Series) -> pd.Series:
        rat = row["rat_norm"]
        if rat == "4G":
            freq_mhz = lte_dl_freq_mhz(row["band"], row["channel_key"])
            band_type = band_type_from_freq_mhz(freq_mhz, "4G")
        elif rat == "5G":
            freq_mhz = parse_float(row["channel_key"])
            band_type = normalize_existing_band_type(row.get("band_type"))
            if band_type is None:
                band_type = band_type_from_freq_mhz(freq_mhz, "5G")
        else:
            freq_mhz = None
            band_type = "unknown"

        return pd.Series(
            {
                "rat_label": "LTE"
                if rat == "4G"
                else ("5G" if rat == "5G" else "unknown"),
                "downlink_freq_mhz": freq_mhz,
                "derived_band_type": band_type,
            }
        )

    classified = work.apply(classify_row, axis=1)
    work = pd.concat([work, classified], axis=1)
    work = work.dropna(subset=["band", "channel_key"])

    summary = (
        work.groupby(
            [
                "rat_label",
                "band",
                "channel_key",
                "downlink_freq_mhz",
                "derived_band_type",
            ],
            dropna=False,
        )
        .agg(
            operators=("operator", join_unique),
            n_operators=("operator", lambda s: len({v for v in s if clean_value(v)})),
            n_cells=("pcell_key", "nunique"),
            n_rows=("pcell_key", "size"),
        )
        .reset_index()
        .rename(columns={"rat_label": "rat", "derived_band_type": "band_type"})
    )

    summary["band_number"] = summary["band"].map(extract_band_number)
    summary["rat_sort"] = summary["rat"].map({"LTE": 0, "5G": 1}).fillna(9)
    summary["band_type_sort"] = (
        summary["band_type"].map(BAND_TYPE_ORDER).fillna(BAND_TYPE_ORDER["unknown"])
    )
    summary["channel_value"] = summary["channel_key"].map(parse_float)
    summary = summary.sort_values(
        [
            "rat_sort",
            "band_type_sort",
            "band_number",
            "downlink_freq_mhz",
            "channel_value",
            "band",
            "channel_key",
        ],
        na_position="last",
    )
    return summary[
        [
            "rat",
            "band",
            "channel_key",
            "downlink_freq_mhz",
            "band_type",
            "operators",
            "n_operators",
            "n_cells",
            "n_rows",
        ]
    ]


def build_band_summary(channel_summary: pd.DataFrame) -> pd.DataFrame:
    work = channel_summary.copy()

    def join_channels(values: Iterable) -> str:
        vals = []
        for v in values:
            vv = clean_value(v)
            if vv is not None:
                vals.append(vv)
        return ",".join(sorted(set(vals), key=lambda x: (parse_float(x) is None, parse_float(x) or 0, x)))

    summary = (
        work.groupby(["rat", "band", "band_type"], dropna=False)
        .agg(
            channel_count=("channel_key", "nunique"),
            channel_keys=("channel_key", join_channels),
            freq_min_mhz=("downlink_freq_mhz", "min"),
            freq_max_mhz=("downlink_freq_mhz", "max"),
            operators=("operators", lambda s: join_unique(",".join(s).split(","))),
            n_operators=("operators", lambda s: len(set(",".join(s).split(",")) - {""})),
            n_cells=("n_cells", "sum"),
            n_rows=("n_rows", "sum"),
        )
        .reset_index()
    )

    summary["band_number"] = summary["band"].map(extract_band_number)
    summary["rat_sort"] = summary["rat"].map({"LTE": 0, "5G": 1}).fillna(9)
    summary["band_type_sort"] = (
        summary["band_type"].map(BAND_TYPE_ORDER).fillna(BAND_TYPE_ORDER["unknown"])
    )
    summary = summary.sort_values(
        ["rat_sort", "band_type_sort", "band_number", "freq_min_mhz", "band"],
        na_position="last",
    )
    return summary[
        [
            "rat",
            "band",
            "band_type",
            "channel_count",
            "channel_keys",
            "freq_min_mhz",
            "freq_max_mhz",
            "operators",
            "n_operators",
            "n_cells",
            "n_rows",
        ]
    ]


def write_band_summaries(
    input_csv: Path, summary_csv: Path, channel_summary_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    channel_summary = build_band_channel_summary(df)
    summary = build_band_summary(channel_summary)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    channel_summary.to_csv(channel_summary_csv, index=False)
    return summary, channel_summary


def load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, path, {"operator", "rat", "pcell_key"})
    df = df.copy()
    df["operator"] = df["operator"].map(normalize_operator)
    df["rat"] = df["rat"].map(normalize_rat)
    df["channel_key"] = df.apply(
        lambda r: extract_channel_key(r["pcell_key"], r["rat"]), axis=1
    )
    return df


def load_band_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(
        df,
        path,
        {
            "operator",
            "rat",
            "band",
            "channel_key",
            "frequency_label",
            "band_type",
        },
    )
    keep_cols = [
        "operator",
        "rat",
        "band",
        "channel_key",
        "frequency_label",
        "band_type",
    ]
    out = df[keep_cols].copy()
    out["operator"] = out["operator"].map(normalize_operator)
    out["rat"] = out["rat"].map(normalize_rat)
    out["band"] = out["band"].map(clean_value)
    out["channel_key"] = out["channel_key"].map(clean_value)
    out["band_type"] = out["band_type"].map(normalize_existing_band_type)
    out = out.dropna(subset=["operator", "rat", "band", "channel_key"])
    out = out.drop_duplicates(
        subset=["operator", "rat", "band", "channel_key"], keep="first"
    )
    return out


def load_area_type(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, path, {"pcell_key", "area_type"})
    out = df.copy()
    if "area_fraction" not in out.columns:
        out["area_fraction"] = 0
    if "n_area_labeled_samples" not in out.columns:
        out["n_area_labeled_samples"] = 0
    out["area_fraction"] = pd.to_numeric(out["area_fraction"], errors="coerce")
    out["n_area_labeled_samples"] = pd.to_numeric(
        out["n_area_labeled_samples"], errors="coerce"
    )
    out = out.sort_values(
        ["pcell_key", "area_fraction", "n_area_labeled_samples"],
        ascending=[True, False, False],
        na_position="last",
    )
    out = out.drop_duplicates(subset=["pcell_key"], keep="first")
    return out[["pcell_key", "area_type"]]


def load_time_period(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, path, {"cell_key", "period"})
    out = df.copy()
    if "period_fraction" not in out.columns:
        out["period_fraction"] = 0
    if "n_samples_in_period" not in out.columns:
        out["n_samples_in_period"] = 0
    out["period_fraction"] = pd.to_numeric(out["period_fraction"], errors="coerce")
    out["n_samples_in_period"] = pd.to_numeric(
        out["n_samples_in_period"], errors="coerce"
    )
    out = out.sort_values(
        ["cell_key", "period_fraction", "n_samples_in_period"],
        ascending=[True, False, False],
        na_position="last",
    )
    out = out.drop_duplicates(subset=["cell_key"], keep="first")
    return out[["cell_key", "period"]].rename(columns={"cell_key": "pcell_key"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    ap.add_argument("--band-csv", type=Path, default=DEFAULT_UE_BAND_CSV)
    ap.add_argument("--area-csv", type=Path, default=DEFAULT_AREA_CSV)
    ap.add_argument("--time-csv", type=Path, default=DEFAULT_TIME_CSV)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--band-summary-csv", type=Path, default=DEFAULT_BAND_SUMMARY_CSV)
    ap.add_argument(
        "--band-channel-summary-csv",
        type=Path,
        default=DEFAULT_BAND_CHANNEL_SUMMARY_CSV,
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only build the band summary from --out-csv; do not rebuild all cells.",
    )
    args = ap.parse_args()

    if not args.summary_only:
        base = load_base(args.base_csv)
        base["band"] = base["band"].map(clean_value)
        base["channel_key"] = base["channel_key"].map(clean_value)
        band_labels = load_band_labels(args.band_csv)
        area = load_area_type(args.area_csv)
        time_period = load_time_period(args.time_csv)

        out = base.merge(
            band_labels,
            on=["operator", "rat", "band", "channel_key"],
            how="left",
            validate="many_to_one",
        )
        out = out.merge(area, on="pcell_key", how="left", validate="many_to_one")
        out = out.merge(time_period, on="pcell_key", how="left", validate="many_to_one")

        # Keep the published report's column order and float precision.
        front = ["operator", "rat", "pcell_key", "band", "frequency_label", "band_type"]
        out = out[front + [col for col in out.columns if col not in front]]

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False, float_format="%.15g")

        print(f"[OK] Wrote {args.out_csv} ({len(out)} rows)")
        print(f"[INFO] base rows: {len(base)}")
        print(f"[INFO] matched frequency_label: {out['frequency_label'].notna().sum()}")
        print(f"[INFO] matched band_type: {out['band_type'].notna().sum()}")
        print(f"[INFO] matched area_type: {out['area_type'].notna().sum()}")
        print(f"[INFO] matched period: {out['period'].notna().sum()}")

    summary, channel_summary = write_band_summaries(
        args.out_csv, args.band_summary_csv, args.band_channel_summary_csv
    )
    print(f"[OK] Wrote {args.band_summary_csv} ({len(summary)} rows)")
    print(
        f"[OK] Wrote {args.band_channel_summary_csv} "
        f"({len(channel_summary)} rows)"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
