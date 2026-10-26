#!/usr/bin/env python3
"""
Build a complete UE-level PCell log from raw operator PCell pickles.

The PCell pickle rows define the output universe: every indexed PCell row is
kept. Timing-advance neighbor logs are left-joined only to annotate
neighbor_count/neighbor_list_raw when a same timestamp + serving cell match
exists, so missing timing rows never drop UE metric samples.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


OPS = ["att", "tmobile", "verizon"]
OP_LABELS = {"att": "A", "tmobile": "T", "verizon": "V"}
OP_FILE_PREFIX = {"A": "att", "T": "tmobile", "V": "verizon"}
FILENAME_RE = re.compile(
    r"^(?P<op>[A-Za-z])_(?P<day>\d+)_.*\.(xlsx|csv)$", re.IGNORECASE
)

UNIX_COL = "Unix TIME_STAMP"
TIME_COL = "TIME_STAMP"
LON_COL = "Lon"
LAT_COL = "Lat"
DIST_COL = "Distance[m]"

LTE_NEI_COL = (
    "Qualcomm Lte/LteAdv Intrafreq Measure PCell Neighbor Cell Neighbor Cell List"
)
NR_NEI_COL = (
    "Qualcomm 5G-NR ML1 Searcher Measurement PCell Neighbor Cell Neighbor Cell List"
)
LTE_PCI_COL = "LTE KPI PCell Serving PCI"
LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"
NR_PCI_COL = "5G KPI PCell RF Serving PCI"
NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"

NR_BAND_COL = "5G KPI PCell RF Band"
LTE_BAND_COL = "LTE KPI PCell Serving Band"
NR_FREQ_COL = "5G KPI PCell RF Frequency [MHz]"

NR_RSRP_COL = "5G KPI PCell RF Serving SS-RSRP [dBm]"
NR_SINR_COL = "5G KPI PCell RF Serving SS-SINR [dB]"
NR_INT_COL = "NR KPI PCell Interference [dBm]"
LTE_RSRP_COL = "LTE KPI PCell Serving RSRP [dBm]"
LTE_SINR_COL = "LTE KPI PCell SINR [dB]"
LTE_INT_COL = "LTE KPI PCell Interference [dBm]"

BAND_TYPE_ORDER = ["low", "mid", "mmwave"]


def clean_value(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    out = out.map(clean_value)
    return out.astype("string")


def normalize_rat_label(rat: Any) -> Optional[str]:
    s = str(rat).strip().upper()
    if s in {"NR", "5G"}:
        return "5G"
    if s in {"LTE", "4G"}:
        return "4G"
    return None


def guess_rat_from_pcell_key(pcell_key: str) -> Optional[str]:
    if str(pcell_key).startswith("NR|"):
        return "5G"
    if str(pcell_key).startswith("LTE|"):
        return "4G"
    if "earfcn=" in str(pcell_key).lower():
        return "4G"
    if "arfcn=" in str(pcell_key).lower():
        return "5G"
    return None


def parse_key_field(pcell_key: Any, field: str) -> Optional[str]:
    m = re.search(rf"(?:^|\|){re.escape(field)}=([^|]+)", str(pcell_key))
    return clean_value(m.group(1)) if m else None


def parse_neighbor_list(val: Any) -> Set[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
        return set()
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return set()
    out: Set[str] = set()
    for part in s.split(","):
        p = clean_value(part)
        if p is not None:
            out.add(p)
    return out


def guess_op_from_timing_name(path: Path) -> Optional[str]:
    m = FILENAME_RE.match(path.name)
    if m:
        return OP_FILE_PREFIX.get(m.group("op").upper())
    name = path.name.lower()
    for op in OPS:
        if op in name:
            return op
    return None


def read_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix != ".csv":
        raise ValueError(f"Unsupported file type: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        width = len(header)
        rows: List[List[str]] = []
        fixed = 0
        first_bad_lineno = None
        for lineno, row in enumerate(reader, start=2):
            if len(row) == width:
                rows.append(row)
                continue
            if len(row) == width + 1 and row[-1] == "":
                rows.append(row[:-1])
            else:
                if first_bad_lineno is None:
                    first_bad_lineno = lineno
                rows.append((row[:width] + [""] * max(0, width - len(row)))[:width])
            fixed += 1
    if fixed:
        warnings.warn(
            f"{path.name}: normalized {fixed} CSV rows "
            f"(first malformed line {first_bad_lineno})",
            RuntimeWarning,
        )
    return pd.DataFrame(rows, columns=header)


def load_timing_for_operator(timing_dir: Path, op: str) -> pd.DataFrame:
    files = sorted(timing_dir.glob("*.csv")) + sorted(timing_dir.glob("*.xlsx"))
    op_files = [p for p in files if guess_op_from_timing_name(p) == op]
    frames = []
    for path in op_files:
        try:
            df = read_input_table(path)
        except Exception as exc:
            print(f"[WARN] skip timing file {path.name}: {exc}")
            continue
        if len(df) > 8:
            df = df.iloc[:-8].copy()
        df["timing_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_timing_neighbor_table(timing_dir: Path, op: str) -> pd.DataFrame:
    timing = load_timing_for_operator(timing_dir, op)
    if timing.empty:
        return pd.DataFrame(
            columns=[
                "rat",
                "unix_timestamp",
                "pcell_pci",
                "timing_channel_id",
                "neighbor_count",
                "neighbor_list_raw",
                "timing_match_count",
                "timing_file",
            ]
        )

    for col in [UNIX_COL, LTE_PCI_COL, LTE_EARFCN_COL, NR_PCI_COL, NR_ARFCN_COL]:
        if col not in timing.columns:
            timing[col] = pd.NA

    timing["_unix"] = pd.to_numeric(timing[UNIX_COL], errors="coerce")
    parts = []
    specs = [
        ("4G", LTE_PCI_COL, LTE_EARFCN_COL, LTE_NEI_COL),
        ("5G", NR_PCI_COL, NR_ARFCN_COL, NR_NEI_COL),
    ]
    for rat, pci_col, chan_col, nei_col in specs:
        if nei_col not in timing.columns:
            continue
        part = pd.DataFrame(
            {
                "rat": rat,
                "unix_timestamp": timing["_unix"],
                "pcell_pci": normalize_id_series(timing[pci_col]),
                "timing_channel_id": normalize_id_series(timing[chan_col]),
                "neighbor_count": timing[nei_col].map(
                    lambda v: int(len(parse_neighbor_list(v)))
                ),
                "neighbor_list_raw": timing[nei_col].map(clean_value),
                "timing_file": timing["timing_file"].map(clean_value),
            }
        )
        part = part.dropna(
            subset=["unix_timestamp", "pcell_pci", "timing_channel_id"]
        )
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    grouped = (
        out.groupby(
            ["rat", "unix_timestamp", "pcell_pci", "timing_channel_id"],
            dropna=False,
            as_index=False,
        )
        .agg(
            neighbor_count=("neighbor_count", "max"),
            neighbor_list_raw=("neighbor_list_raw", "first"),
            timing_match_count=("neighbor_count", "size"),
            timing_file=("timing_file", "first"),
        )
    )
    return grouped


def extract_band_number(band: Any) -> Optional[int]:
    s = str(band).strip()
    m = re.search(r"\bBand\s*(\d+)\b", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bb(\d+)\b", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def lte_dl_freq_mhz(band: int, earfcn: int) -> Optional[float]:
    band_table = {
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
    if band not in band_table:
        return None
    f_low, n_off, n_min, n_max = band_table[band]
    if not (n_min <= earfcn <= n_max):
        return None
    return f_low + 0.1 * (earfcn - n_off)


def safe_lte_dl_freq_mhz(band: Any, earfcn: Any) -> Optional[float]:
    band_num = extract_band_number(band)
    if band_num is None:
        return None
    try:
        earfcn_int = int(float(str(earfcn)))
    except Exception:
        return None
    return lte_dl_freq_mhz(band_num, earfcn_int)


def nr_arfcn_to_freq_mhz(arfcn: Any) -> Optional[float]:
    try:
        n_ref = int(round(float(arfcn)))
    except Exception:
        return None
    if 0 <= n_ref <= 599999:
        return 0.005 * n_ref
    if 600000 <= n_ref <= 2016666:
        return 3000.0 + 0.015 * (n_ref - 600000)
    if 2016667 <= n_ref <= 3279165:
        return 24250.08 + 0.06 * (n_ref - 2016667)
    return None


def freq_to_key(freq_mhz: Any) -> Optional[str]:
    try:
        freq = float(freq_mhz)
    except Exception:
        return None
    if not np.isfinite(freq):
        return None
    return f"{freq:.2f}".rstrip("0").rstrip(".")


def infer_band_type(rat: Any, freq_mhz: Any) -> Optional[str]:
    try:
        freq = float(freq_mhz)
    except Exception:
        return None
    if not np.isfinite(freq):
        return None
    if freq < 1000:
        return "low"
    if str(rat) == "5G" and freq >= 24000:
        return "mmwave"
    return "mid"


def normalize_band_type(value: Any) -> Optional[str]:
    s = clean_value(value)
    if s is None:
        return None
    s = s.lower()
    if s in {"low", "mid", "mmwave"}:
        return s
    if s in {"mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return None


def load_band_type_maps(
    path: Path,
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    required = {"rat", "band", "band_type", "channel_keys"}
    if not required.issubset(df.columns):
        return {}, {}
    by_band: Dict[Tuple[str, str], str] = {}
    by_channel: Dict[Tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        rat = normalize_rat_label(getattr(row, "rat"))
        band = clean_value(getattr(row, "band"))
        band_type = normalize_band_type(getattr(row, "band_type"))
        if rat is None or band_type is None:
            continue
        if band is not None:
            by_band[(rat, band)] = band_type
        for channel in str(getattr(row, "channel_keys")).split(","):
            channel = clean_value(channel)
            if channel is not None:
                by_channel[(rat, channel)] = band_type
    return by_band, by_channel


def load_channel_band_map(path: Path) -> Dict[Tuple[str, str], str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"rat", "band", "channel_keys"}
    if not required.issubset(df.columns):
        return {}
    out: Dict[Tuple[str, str], str] = {}
    for row in df.itertuples(index=False):
        rat = normalize_rat_label(getattr(row, "rat"))
        band = clean_value(getattr(row, "band"))
        if rat is None or band is None:
            continue
        for channel in str(getattr(row, "channel_keys")).split(","):
            channel = clean_value(channel)
            if channel is not None:
                out[(rat, channel)] = band
    return out


def compact_frequency_label(label: Any) -> str:
    return re.sub(r"\s+", "", str(label).strip())


def add_operator_local_frequency_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    channels = (
        out[["operator", "rat", "band", "channel_key", "channel_value_num"]]
        .dropna(subset=["operator", "rat", "band", "channel_key"])
        .drop_duplicates()
        .copy()
    )
    channels["channel_value_num"] = pd.to_numeric(
        channels["channel_value_num"], errors="coerce"
    )
    channels = channels.sort_values(
        ["operator", "rat", "band", "channel_value_num", "channel_key"],
        na_position="last",
    )
    channels["freq_idx"] = channels.groupby(["operator", "rat", "band"]).cumcount() + 1
    channels["frequency_label"] = channels.apply(
        lambda r: f"{r.band}-f{int(r.freq_idx)}", axis=1
    )
    label_map = {
        (str(r.operator), str(r.rat), str(r.band), str(r.channel_key)): str(r.frequency_label)
        for r in channels.itertuples(index=False)
    }
    out["frequency_label"] = out.apply(
        lambda r: label_map.get(
            (str(r["operator"]), str(r["rat"]), str(r["band"]), str(r["channel_key"]))
        ),
        axis=1,
    )
    return out


def choose_numeric(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(df[col], errors="coerce")


def choose_text(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=index, dtype="object")
    return df[col].map(clean_value)


def build_pcell_row_table(
    df: pd.DataFrame, pcell_index: Dict[str, Iterable[int]]
) -> pd.DataFrame:
    needed_cols = [
        TIME_COL,
        UNIX_COL,
        LON_COL,
        LAT_COL,
        DIST_COL,
        NR_BAND_COL,
        LTE_BAND_COL,
        NR_FREQ_COL,
        NR_RSRP_COL,
        NR_SINR_COL,
        NR_INT_COL,
        LTE_RSRP_COL,
        LTE_SINR_COL,
        LTE_INT_COL,
    ]
    cols = [c for c in needed_cols if c in df.columns]
    row_arrays = []
    key_arrays = []
    for pcell_key, rows in pcell_index.items():
        arr = np.asarray(rows, dtype=int)
        if len(arr) == 0:
            continue
        row_arrays.append(arr)
        key_arrays.append(np.full(len(arr), pcell_key, dtype=object))
    if not row_arrays:
        return pd.DataFrame(columns=cols + ["pcell_key", "pkl_row_index"])

    row_positions = np.concatenate(row_arrays)
    pcell_keys = np.concatenate(key_arrays)
    is_default_range = (
        isinstance(df.index, pd.RangeIndex)
        and df.index.start == 0
        and df.index.step == 1
    )
    if is_default_range:
        base = df.iloc[row_positions][cols].copy()
        pkl_row_index = row_positions
    else:
        base = df.loc[row_positions, cols].copy()
        pkl_row_index = base.index.to_numpy()
    base = base.reset_index(drop=True)
    base["pcell_key"] = pcell_keys
    base["pkl_row_index"] = pkl_row_index
    return base


def median_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else np.nan


def first_clean_value(series: pd.Series) -> Optional[str]:
    for value in series:
        cleaned = clean_value(value)
        if cleaned is not None:
            return cleaned
    return None


def build_operator_records(
    pkl_path: Path,
    timing_dir: Path,
    band_summary_csv: Path,
) -> pd.DataFrame:
    payload = pd.read_pickle(pkl_path)
    df = payload["df"]
    pcell_index = payload["pcell_index"]
    meta = payload.get("meta", {})
    op = str(meta.get("operator") or pkl_path.stem.replace("_pcell", "")).lower()

    base = build_pcell_row_table(df, pcell_index)
    out = pd.DataFrame(index=base.index)
    out["operator"] = op
    out["pcell_key"] = base["pcell_key"].astype(str)
    out["pkl_row_index"] = base["pkl_row_index"].astype(int)
    out["rat"] = out["pcell_key"].map(guess_rat_from_pcell_key)
    out["time_stamp"] = choose_text(base, TIME_COL, base.index)
    out["unix_timestamp"] = choose_numeric(base, UNIX_COL, base.index)
    out["lon"] = choose_numeric(base, LON_COL, base.index)
    out["lat"] = choose_numeric(base, LAT_COL, base.index)
    out["distance_m"] = choose_numeric(base, DIST_COL, base.index)

    is_nr = out["rat"].eq("5G")
    is_lte = out["rat"].eq("4G")
    out["band"] = pd.NA
    out.loc[is_nr, "band"] = choose_text(base, NR_BAND_COL, base.index).loc[is_nr]
    out.loc[is_lte, "band"] = choose_text(base, LTE_BAND_COL, base.index).loc[is_lte]
    band_fill = out.groupby("pcell_key")["band"].transform(first_clean_value)
    out["band"] = out["band"].fillna(band_fill)

    out["pcell_pci"] = out["pcell_key"].map(lambda k: parse_key_field(k, "pci"))
    out["pcell_arfcn"] = out["pcell_key"].map(lambda k: parse_key_field(k, "arfcn"))
    out["pcell_earfcn"] = out["pcell_key"].map(lambda k: parse_key_field(k, "earfcn"))
    out["pcell_freq_from_key_mhz"] = out["pcell_key"].map(
        lambda k: parse_key_field(k, "freq")
    )

    out["rsrp"] = np.nan
    out["sinr"] = np.nan
    out["interference"] = np.nan
    out.loc[is_nr, "rsrp"] = choose_numeric(base, NR_RSRP_COL, base.index).loc[is_nr]
    out.loc[is_nr, "sinr"] = choose_numeric(base, NR_SINR_COL, base.index).loc[is_nr]
    out.loc[is_nr, "interference"] = choose_numeric(base, NR_INT_COL, base.index).loc[is_nr]
    out.loc[is_lte, "rsrp"] = choose_numeric(base, LTE_RSRP_COL, base.index).loc[is_lte]
    out.loc[is_lte, "sinr"] = choose_numeric(base, LTE_SINR_COL, base.index).loc[is_lte]
    out.loc[is_lte, "interference"] = choose_numeric(base, LTE_INT_COL, base.index).loc[is_lte]

    out["channel_key"] = pd.NA
    out.loc[is_lte, "channel_key"] = out.loc[is_lte, "pcell_earfcn"]
    nr_freq = choose_numeric(base, NR_FREQ_COL, base.index).map(freq_to_key)
    out.loc[is_nr, "channel_key"] = out.loc[is_nr, "pcell_freq_from_key_mhz"].fillna(
        nr_freq.loc[is_nr]
    )
    missing_nr_channel = is_nr & out["channel_key"].isna()
    out.loc[missing_nr_channel, "channel_key"] = out.loc[
        missing_nr_channel, "pcell_arfcn"
    ].map(lambda v: freq_to_key(nr_arfcn_to_freq_mhz(v)))

    out["frequency_mhz"] = np.nan
    out.loc[is_nr, "frequency_mhz"] = pd.to_numeric(
        out.loc[is_nr, "channel_key"], errors="coerce"
    )
    out.loc[is_lte, "frequency_mhz"] = [
        safe_lte_dl_freq_mhz(band, earfcn)
        for band, earfcn in zip(out.loc[is_lte, "band"], out.loc[is_lte, "pcell_earfcn"])
    ]
    out["channel_value_num"] = pd.to_numeric(out["channel_key"], errors="coerce")

    channel_band_map = load_channel_band_map(band_summary_csv)
    missing_band = out["band"].isna() & out["channel_key"].notna()
    if missing_band.any():
        out.loc[missing_band, "band"] = [
            channel_band_map.get((str(rat), str(channel)))
            for rat, channel in zip(
                out.loc[missing_band, "rat"],
                out.loc[missing_band, "channel_key"],
            )
        ]

    by_band, by_channel = load_band_type_maps(band_summary_csv)
    band_types = []
    for rat, band, channel, freq in zip(
        out["rat"], out["band"], out["channel_key"], out["frequency_mhz"]
    ):
        band_type = by_band.get((str(rat), str(band)))
        if band_type is None:
            band_type = by_channel.get((str(rat), str(channel)))
        if band_type is None:
            band_type = infer_band_type(rat, freq)
        band_types.append(band_type)
    out["band_type"] = band_types
    out = add_operator_local_frequency_labels(out)

    area_map = meta.get("pcell_area_type") or {}
    out["area_type"] = out["pcell_key"].map(area_map).map(lambda v: clean_value(v) or "unknown")
    out["band_type_group_label"] = out.apply(
        lambda r: (
            f"{compact_frequency_label(r['frequency_label'])}-"
            f"{OP_LABELS.get(str(r['operator']).lower(), str(r['operator'])[:1].upper())}"
        )
        if clean_value(r.get("frequency_label")) is not None
        else pd.NA,
        axis=1,
    )

    timing = build_timing_neighbor_table(timing_dir, op)
    out["timing_channel_id"] = pd.NA
    out.loc[is_lte, "timing_channel_id"] = out.loc[is_lte, "pcell_earfcn"]
    out.loc[is_nr, "timing_channel_id"] = out.loc[is_nr, "pcell_arfcn"]
    out["pcell_pci"] = out["pcell_pci"].map(clean_value).astype("string")
    out["timing_channel_id"] = out["timing_channel_id"].map(clean_value).astype("string")

    if not timing.empty:
        out = out.merge(
            timing,
            on=["rat", "unix_timestamp", "pcell_pci", "timing_channel_id"],
            how="left",
            validate="many_to_one",
        )
    else:
        out["neighbor_count"] = np.nan
        out["neighbor_list_raw"] = pd.NA
        out["timing_match_count"] = np.nan
        out["timing_file"] = pd.NA
    out["has_neighbor_match"] = out["neighbor_count"].notna()
    out["pkl"] = pkl_path.name

    return out[
        [
            "operator",
            "rat",
            "pcell_key",
            "pkl_row_index",
            "time_stamp",
            "unix_timestamp",
            "lon",
            "lat",
            "distance_m",
            "area_type",
            "band",
            "band_type",
            "pcell_pci",
            "pcell_arfcn",
            "pcell_earfcn",
            "pcell_freq_from_key_mhz",
            "channel_key",
            "channel_value_num",
            "frequency_mhz",
            "frequency_label",
            "band_type_group_label",
            "rsrp",
            "interference",
            "sinr",
            "neighbor_count",
            "has_neighbor_match",
            "timing_match_count",
            "neighbor_list_raw",
            "timing_file",
            "pkl",
        ]
    ]


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "operator",
        "rat",
        "band_type",
        "band",
        "channel_key",
        "frequency_label",
        "band_type_group_label",
        "area_type",
    ]
    rows = []
    for key, g in records.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        matched = pd.to_numeric(g["neighbor_count"], errors="coerce").dropna()
        row.update(
            {
                "number_of_cells": int(g["pcell_key"].nunique()),
                "record_count": int(len(g)),
                "neighbor_matched_records": int(g["has_neighbor_match"].sum()),
                "neighbor_match_rate": float(g["has_neighbor_match"].mean()),
                "ue_avg_interferers": float(matched.mean()) if len(matched) else np.nan,
                "ue_interferers_std": float(matched.std(ddof=1)) if len(matched) > 1 else np.nan,
                "median_rsrp": median_numeric(g["rsrp"]),
                "median_interference": median_numeric(g["interference"]),
                "median_sinr": median_numeric(g["sinr"]),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["operator_sort"] = summary["operator"].map(
        {op: i for i, op in enumerate(OPS)}
    )
    summary["rat_sort"] = summary["rat"].map({"4G": 0, "5G": 1})
    summary["band_type_sort"] = summary["band_type"].map(
        {name: i for i, name in enumerate(BAND_TYPE_ORDER)}
    )
    summary["channel_value_num"] = pd.to_numeric(summary["channel_key"], errors="coerce")
    return summary.sort_values(
        [
            "operator_sort",
            "rat_sort",
            "band_type_sort",
            "band",
            "channel_value_num",
            "frequency_label",
            "area_type",
        ],
        na_position="last",
    ).drop(columns=["operator_sort", "rat_sort", "band_type_sort", "channel_value_num"])


def find_pcell_pickle(pkl_root: Path, op: str) -> Optional[Path]:
    exact = pkl_root / f"{op}_pcell.pkl"
    if exact.exists():
        return exact
    matches = sorted(pkl_root.rglob(f"*{op}*_pcell.pkl"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-root", type=Path, default=ROOT / "data/intermediate_data/2024/pkl_output")
    ap.add_argument("--timing-dir", type=Path, default=ROOT / "data/raw_data/2024/timing_advance")
    ap.add_argument("--band-summary", type=Path, default=ROOT / "data/intermediate_data/2024/all_cells_band_summary.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/intermediate_data/2024")
    ap.add_argument("--operators", nargs="+", default=OPS)
    ap.add_argument("--write-operator-reports", action="store_true",
                    help="Also write a separate UE report for each operator")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = args.out_dir / "complete_ue_level_metrics_report.csv"
    summary_csv = args.out_dir / "complete_ue_level_summary.csv"

    summaries = []
    wrote_report = False
    for op in args.operators:
        pkl_path = find_pcell_pickle(args.pkl_root, op)
        if pkl_path is None:
            print(f"[WARN] No pcell pickle for {op}", flush=True)
            continue
        print(f"[INFO] Building complete UE log for {op} from {pkl_path}", flush=True)
        records = build_operator_records(pkl_path, args.timing_dir, args.band_summary)
        records.to_csv(report_csv, index=False, mode="a" if wrote_report else "w", header=not wrote_report)
        wrote_report = True
        if args.write_operator_reports:
            op_csv = args.out_dir / f"complete_ue_level_metrics_report_{op}.csv"
            records.to_csv(op_csv, index=False)
        summaries.append(summarize(records))
        print(
            f"[OK] {op}: rows={len(records)}, cells={records['pcell_key'].nunique()}, "
            f"neighbor_match_rate={records['has_neighbor_match'].mean():.3f}",
            flush=True,
        )

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(summary_csv, index=False)
        print(f"[OK] wrote {report_csv}", flush=True)
        print(f"[OK] wrote {summary_csv}", flush=True)
    else:
        raise SystemExit("No records produced")


if __name__ == "__main__":
    main()
