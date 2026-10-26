#!/usr/bin/env python3
"""Build the 2022 per-cell data from the released XCAL CSVs.

This is the single entry point for the operator split and per-cell cleaning.
The operator CSVs are temporary; the run log and per-cell CSVs are retained.
Device IDs in the released inputs and device map must use the same aliases.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]

PHONE_OP: dict[str, str] = {}
OPER_NORM = {
    "Verizon ": "Verizon",
    "Verizon": "Verizon",
    "AT&T": "AT&T",
    "T-Mobile": "T-Mobile",
}
UNRELIABLE_OP = {"No service", "(Searching)"}


def normalize_phone(v) -> Optional[str]:
    """Normalize an anonymized device alias (or legacy numeric ID)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s.lower() in ("", "unknown", "nan"):
        return None
    if s.lower().startswith("device_"):
        return s.lower()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    return s[-10:] if len(s) >= 10 else (s or None)


def detect_by_phone(series: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    cleaned = [normalize_phone(v) for v in series.dropna()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None, None
    phone = Counter(cleaned).most_common(1)[0][0]
    return phone, PHONE_OP.get(phone)


def detect_by_operator(series: pd.Series) -> Optional[str]:
    vals = [str(v).strip() if not pd.isna(v) else None for v in series]
    vals = [v for v in vals if v and v not in UNRELIABLE_OP]
    if not vals:
        return None
    top = Counter(vals).most_common(1)[0][0]
    return OPER_NORM.get(top, top)


def split_on_gap(df: pd.DataFrame, gap_s: float) -> List[pd.DataFrame]:
    diffs = df["Timestamp"].diff().fillna(0)
    run_id = diffs.gt(gap_s).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(run_id, sort=False)]


def first_last_time_stamp(run: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Return first/last non-null TIME_STAMP within the run."""
    if "TIME_STAMP" not in run.columns:
        return None, None
    ts = run["TIME_STAMP"].dropna()
    if ts.empty:
        return None, None
    return str(ts.iloc[0]), str(ts.iloc[-1])


def dedup_by_coverage(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]:
    """
    First-come-first-served by run-start.
    Drop rows whose Timestamp is <= the max Timestamp already covered by an
    earlier-processed run. Tiebreak runs of equal start by more rows wins.
    Returns (out, runs_kept_whole, runs_trimmed, runs_dropped).
    """
    runs = (
        df.groupby(["source_file", "run_idx"])["Timestamp"]
        .agg(rmin="min", rmax="max", rcount="count")
        .reset_index()
        .sort_values(["rmin", "rcount"], ascending=[True, False])
        .reset_index(drop=True)
    )
    keep_parts = []
    covered_until = -float("inf")
    runs_kept = runs_trimmed = runs_dropped = 0
    for _, r in runs.iterrows():
        sub = df[(df["source_file"] == r["source_file"]) & (df["run_idx"] == r["run_idx"])]
        new_rows = sub[sub["Timestamp"] > covered_until]
        if len(new_rows) == 0:
            runs_dropped += 1
        elif len(new_rows) == len(sub):
            runs_kept += 1
        else:
            runs_trimmed += 1
        if len(new_rows):
            keep_parts.append(new_rows)
        covered_until = max(covered_until, r["rmax"])
    out = (
        pd.concat(keep_parts, ignore_index=True)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    # final sort to put runs in start-time order with rows time-sorted within
    run_starts = out.groupby(["source_file", "run_idx"])["Timestamp"].transform("min")
    out = (
        out.assign(_start=run_starts)
        .sort_values(["_start", "Timestamp"])
        .drop(columns="_start")
        .reset_index(drop=True)
    )
    return out, runs_kept, runs_trimmed, runs_dropped



def split_merged_by_operator(merged_dir: Path, out_dir: Path, gap_s: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = {}             # operator -> list of dataframes
    log_rows: list[dict] = []
    stats = Counter()
    conflicts = 0

    files = sorted(
        f for f in os.listdir(merged_dir) if f.endswith(".csv") and not f.startswith(".")
    )
    print(f"Processing {len(files)} merged files...")

    for fn in files:
        df = pd.read_csv(merged_dir / fn, low_memory=False)
        df = df.sort_values("Timestamp").reset_index(drop=True)
        runs = split_on_gap(df, gap_s=gap_s)

        for run_idx, run in enumerate(runs):
            op_from_op = (
                detect_by_operator(run["Smart Phone System Info Operator"])
                if "Smart Phone System Info Operator" in run.columns
                else None
            )
            ph, op_from_ph = (
                detect_by_phone(run["Smart Phone Mobile Info Line1 Number"])
                if "Smart Phone Mobile Info Line1 Number" in run.columns
                else (None, None)
            )

            if op_from_op and op_from_ph and op_from_op != op_from_ph:
                conflicts += 1
                chosen = op_from_ph
                op_source = "phone (conflict, phone wins)"
                stats["conflict_phone_wins"] += 1
            elif op_from_ph:
                chosen = op_from_ph
                op_source = "phone"
                stats["by_phone"] += 1
            elif op_from_op:
                chosen = op_from_op
                op_source = "operator"
                stats["by_operator"] += 1
            else:
                chosen = "unknown"
                op_source = "unknown"
                stats["unknown"] += 1

            t_start, t_end = first_last_time_stamp(run)
            log_rows.append({
                "source_file": fn,
                "run_idx": run_idx,
                "n_rows": len(run),
                "start_TIME_STAMP": t_start,
                "end_TIME_STAMP": t_end,
                "start_Timestamp": float(run["Timestamp"].iloc[0]),
                "end_Timestamp": float(run["Timestamp"].iloc[-1]),
                "operator": chosen,
                "op_source": op_source,
            })

            tagged = run.copy()
            tagged["source_file"] = fn
            tagged["run_idx"] = run_idx
            buckets.setdefault(chosen, []).append(tagged)

    # Write the run log
    log_df = pd.DataFrame(log_rows)
    log_path = out_dir / "operator_runs_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n[OK] wrote run log -> {log_path}  ({len(log_df):,} runs)")

    # Write per-operator csvs
    print("\nOperator breakdown:")
    for op, dfs in sorted(buckets.items()):
        out = pd.concat(dfs, ignore_index=True)
        out, n_kept, n_trim, n_drop = dedup_by_coverage(out)
        fname = op.replace("/", "_").replace(" ", "_") + ".csv"
        path = out_dir / fname
        out.to_csv(path, index=False)
        print(
            f"  {op:10s}  runs={len(dfs):>4d}  rows_after_dedup={len(out):>9,}  "
            f"(kept_whole={n_kept}, trimmed={n_trim}, dropped={n_drop})  -> {path}  "
            f"({os.path.getsize(path)/1024/1024:.1f} MB)"
        )

    print(f"\nDetection stats: {dict(stats)}")
    print(f"Conflicts (operator vs device ID disagreed; device ID won): {conflicts}")


# ---------------------------------------------------------------------------
# Column names (2022 schema after merge + Sheet2 filter)
# ---------------------------------------------------------------------------
TS_COL = "Timestamp"
LON_COL = "Lon"
LAT_COL = "Lat"
EVENT_TECH_COL = "Event Technology"
SRC_COL = "source_file"
RUN_COL = "run_idx"

NR_PCI_COL = "5G KPI PCell RF Serving PCI"
NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"
NR_FREQ_COL = "5G KPI PCell RF Frequency [MHz]"
NR_BAND_COL = "5G KPI PCell RF Band"
NR_RSRP_COL = "5G KPI PCell RF Serving SS-RSRP [dBm]"
NR_SINR_COL = "5G KPI PCell RF Serving SS-SINR [dB]"

LTE_PCI_COL = "LTE KPI PCell Serving PCI"
LTE_BAND_COL = "LTE KPI PCell Serving Band"
LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"
# NOTE: 2022 csv-side (Sheet2-filtered) columns use no space before "[dBm]"/"[dB]".
LTE_RSRP_COL = "LTE KPI PCell Serving RSRP[dBm]"
LTE_SINR_COL = "LTE KPI PCell SINR[dB]"
LTE_RSRQ_COL = "LTE KPI PCell Serving RSRQ[dB]"

NR_NEI_COL = (
    "Qualcomm 5G-NR ML1 Searcher Measurement PCell Neighbor Cell Neighbor Cell List"
)
LTE_NEI_COL = (
    "Qualcomm Lte/LteAdv Intrafreq Measure PCell Neighbor Cell Neighbor Cell List"
)

NR_PREFIX = "5G KPI "
LTE_PREFIX = "LTE KPI "
ROLE_RE = re.compile(r"(?P<role>PCell|SCell\[(?P<idx>\d+)\])", re.IGNORECASE)

BAND_TYPE_ORDER = ["low", "mid", "mmwave"]


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_value(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def to_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[col].astype("string")


# ---------------------------------------------------------------------------
# Component discovery + interference columns (from data_cleaning_modified_v6.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentCols:
    rat: str  # "NR" or "LTE"
    role: str  # "PCell" or "SCell"
    idx: Optional[int]
    pci_col: str
    rsrp_col: str
    sinr_col: str


def discover_components(df: pd.DataFrame) -> List[ComponentCols]:
    """Find every PCell/SCell component that has PCI + RSRP + SINR columns."""
    bucket: Dict[Tuple[str, str, Optional[int]], Dict[str, str]] = {}
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith(NR_PREFIX):
            rat = "NR"
            body = col_str[len(NR_PREFIX):]
        elif col_str.startswith(LTE_PREFIX):
            rat = "LTE"
            body = col_str[len(LTE_PREFIX):]
        else:
            continue
        rm = ROLE_RE.search(body)
        if not rm:
            continue
        role_raw = rm.group("role")
        idx = rm.group("idx")
        if role_raw.lower().startswith("pcell"):
            role, sc_idx = "PCell", None
        else:
            role, sc_idx = "SCell", int(idx) if idx is not None else None

        # PCI: must END with "Serving PCI" so we exclude PCI-SSB Idx, PCI MOD 3, etc.
        if re.search(r"Serving PCI$", col_str, re.IGNORECASE):
            kind = "pci"
        elif re.search(r"RSRP", col_str, re.IGNORECASE):
            kind = "rsrp"
        elif re.search(r"SINR", col_str, re.IGNORECASE):
            kind = "sinr"
        else:
            continue
        bucket.setdefault((rat, role, sc_idx), {})[kind] = col_str

    comps: List[ComponentCols] = []
    for (rat, role, sc_idx), d in bucket.items():
        if {"pci", "rsrp", "sinr"}.issubset(d.keys()):
            comps.append(ComponentCols(rat, role, sc_idx, d["pci"], d["rsrp"], d["sinr"]))
    comps.sort(key=lambda c: (0 if c.rat == "NR" else 1,
                              0 if c.role == "PCell" else 1,
                              c.idx if c.idx is not None else -1))
    return comps


def add_interference_columns(df: pd.DataFrame, comps: List[ComponentCols]) -> Dict[Tuple[str, str, Optional[int]], str]:
    """For each component, add `<RAT> KPI {PCell|SCell[N]} Interference [dBm]` = RSRP - SINR."""
    name_map: Dict[Tuple[str, str, Optional[int]], str] = {}
    for c in comps:
        name = (f"{c.rat} KPI PCell Interference [dBm]" if c.role == "PCell"
                else f"{c.rat} KPI SCell[{c.idx}] Interference [dBm]")
        rsrp = pd.to_numeric(df[c.rsrp_col], errors="coerce")
        sinr = pd.to_numeric(df[c.sinr_col], errors="coerce")
        df[name] = (rsrp - sinr).where(rsrp.notna() & sinr.notna(), other=np.nan)
        name_map[(c.rat, c.role, c.idx)] = name
    return name_map


# ---------------------------------------------------------------------------
# Row filter (same rule as reference drop_rows_with_no_signal)
# ---------------------------------------------------------------------------
def drop_rows_with_no_signal(df: pd.DataFrame) -> pd.DataFrame:
    nr_pci = to_num(df, NR_PCI_COL)
    lte_pci = to_num(df, LTE_PCI_COL)
    nr_rsrp = to_num(df, NR_RSRP_COL)
    nr_sinr = to_num(df, NR_SINR_COL)
    lte_rsrp = to_num(df, LTE_RSRP_COL)
    lte_rsrq = to_num(df, LTE_RSRQ_COL)

    nr_present = nr_pci.notna()
    lte_present = lte_pci.notna()
    nr_complete = nr_present & nr_rsrp.notna() & nr_sinr.notna()
    lte_complete = lte_present & lte_rsrp.notna() & lte_rsrq.notna()

    keep_only_nr = nr_present & (~lte_present) & nr_complete
    keep_only_lte = (~nr_present) & lte_present & lte_complete
    keep_both = nr_present & lte_present & (nr_complete | lte_complete)
    return df.loc[keep_only_nr | keep_only_lte | keep_both].copy()


# ---------------------------------------------------------------------------
# Cell-key builders with instance-level disambiguation by distance.
#
# Same base key (PCI + ARFCN/EARFCN) seen on segments that are more than
# `threshold_km` apart is treated as a distinct cell instance — operators reuse
# PCI+ARFCN across geographically separated cells. v6's threshold is keyed by
# (rat, NR band group, area_type) from data/threshold.csv. 2022 lacks area_type
# coverage (data/area.csv only spans Oct–Nov 2024), so we collapse the area axis
# and use the 'suburban' row as a fixed per-tech constant.
# ---------------------------------------------------------------------------
CELL_DISTANCE_THRESHOLDS_KM: Dict[Tuple[str, Optional[str]], float] = {
    ("LTE", None):     20.0,
    ("NR",  "low"):    20.0,
    ("NR",  "mid"):    15.0,
    ("NR",  "mmwave"):  0.3,
}
DEFAULT_CELL_DISTANCE_KM = 20.0

# Threshold for treating a single-row GPS step as a spike (~96.6 m). Same value
# as scripts/data_cleaning_modified_v6.py's `GPS_JUMP_MILES * 1.609344`.
GPS_JUMP_KM = 0.06 * 1.609344

_NR_BAND_GROUP_MAP = {
    "n2": "low", "n5": "low", "n12": "low", "n13": "low", "n14": "low",
    "n25": "low", "n26": "low", "n29": "low", "n30": "low", "n66": "low", "n71": "low",
    "n41": "mid", "n48": "mid", "n77": "mid", "n78": "mid", "n79": "mid",
    "n257": "mmwave", "n258": "mmwave", "n260": "mmwave", "n261": "mmwave", "n262": "mmwave",
}


def _nr_band_to_group(band) -> Optional[str]:
    if band is None or (isinstance(band, float) and pd.isna(band)) or pd.isna(band):
        return None
    s = str(band).strip().lower()
    m = (re.search(r"\bn\s*(\d+)\b", s)
         or re.search(r"\bband\s*(\d+)\b", s)
         or re.search(r"\bb(\d+)\b", s))
    if m:
        return _NR_BAND_GROUP_MAP.get(f"n{m.group(1)}")
    if s in {"low", "mid"}:
        return s
    if s in {"mmwave", "mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return None


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    if any(pd.isna(v) for v in (lon1, lat1, lon2, lat2)):
        return float("nan")
    rlon1, rlat1, rlon2, rlat2 = map(
        math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)]
    )
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


def _lookup_cell_threshold_km(rat: str, nr_band_group: Optional[str]) -> float:
    if rat == "LTE":
        return CELL_DISTANCE_THRESHOLDS_KM.get(("LTE", None), DEFAULT_CELL_DISTANCE_KM)
    return CELL_DISTANCE_THRESHOLDS_KM.get(("NR", nr_band_group), DEFAULT_CELL_DISTANCE_KM)


def _filter_gps_jumps(
    lon: np.ndarray, lat: np.ndarray, boundary_change: List[bool]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Hold previous output position when consecutive raw GPS steps exceed GPS_JUMP_KM.

    Resets state at (source_file, run_idx) boundaries so the last point of one
    run cannot trigger a hold at the first point of the next.
    Returns (lon_filtered, lat_filtered, n_held). Mirrors v6's filter.
    """
    n = len(lon)
    lon_f = lon.astype(float).copy()
    lat_f = lat.astype(float).copy()
    prev_raw_lon = float("nan")
    prev_raw_lat = float("nan")
    prev_out_lon = float("nan")
    prev_out_lat = float("nan")
    n_held = 0
    for i in range(n):
        lo = lon[i]
        la = lat[i]
        if (
            boundary_change[i]
            or pd.isna(prev_raw_lon) or pd.isna(prev_raw_lat)
            or pd.isna(lo) or pd.isna(la)
        ):
            lon_f[i] = lo
            lat_f[i] = la
        else:
            d = _haversine_km(prev_raw_lon, prev_raw_lat, lo, la)
            if not math.isnan(d) and d > GPS_JUMP_KM:
                lon_f[i] = prev_out_lon
                lat_f[i] = prev_out_lat
                n_held += 1
            else:
                lon_f[i] = lo
                lat_f[i] = la
        prev_raw_lon = lo
        prev_raw_lat = la
        prev_out_lon = lon_f[i]
        prev_out_lat = lat_f[i]
    return lon_f, lat_f, n_held


def build_cell_keys(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (nr_cell_key, lte_cell_key) with cell-instance disambiguation.

    Builds per-RAT segments (contiguous runs of identical base key within one
    (source_file, run_idx) boundary), then for each segment compares its start
    point against every prior segment-end point of the same base key. Closest
    point within `threshold_km` wins; otherwise a new instance is opened. Keys
    are suffixed with `|inst=<N>` so different physical cells with the same
    PCI+ARFCN end up as different `cell_key` values.
    """
    nr_pci = to_str(df, NR_PCI_COL).map(clean_value)
    nr_arfcn = to_str(df, NR_ARFCN_COL).map(clean_value)
    nr_freq = to_str(df, NR_FREQ_COL).map(clean_value)
    nr_band = to_str(df, NR_BAND_COL).map(clean_value)
    lte_pci = to_str(df, LTE_PCI_COL).map(clean_value)
    lte_earfcn = to_str(df, LTE_EARFCN_COL).map(clean_value)

    lon_raw = to_num(df, LON_COL).to_numpy()
    lat_raw = to_num(df, LAT_COL).to_numpy()
    src = to_str(df, SRC_COL).fillna("").tolist()
    run = to_str(df, RUN_COL).fillna("").tolist()
    n = len(df)

    boundary_change = [True] + [
        (src[i], run[i]) != (src[i - 1], run[i - 1]) for i in range(1, n)
    ]
    lon, lat, n_gps_held = _filter_gps_jumps(lon_raw, lat_raw, boundary_change)
    if n_gps_held:
        print(
            f"    GPS-jump filter: held previous position for {n_gps_held:,} of {n:,} rows "
            f"(threshold = {GPS_JUMP_KM*1000:.0f} m)",
            flush=True,
        )

    def _nr_base_key(i: int) -> Optional[str]:
        v = nr_pci.iat[i]
        if v is None:
            return None
        return (
            f"NR|PCell|pci={v}"
            f"|arfcn={nr_arfcn.iat[i] or ''}"
            f"|freq={nr_freq.iat[i] or ''}"
        )

    def _lte_base_key(i: int) -> Optional[str]:
        v = lte_pci.iat[i]
        if v is None:
            return None
        return f"LTE|PCell|pci={v}|earfcn={lte_earfcn.iat[i] or ''}"

    def _build_segments(base_key_fn, rat: str) -> List[dict]:
        segs: List[dict] = []
        for i in range(n):
            bk = base_key_fn(i)
            if bk is None:
                continue
            boundary = (src[i], run[i])
            if (not segs) or segs[-1]["key"] != bk or segs[-1]["boundary"] != boundary:
                segs.append({
                    "key": bk,
                    "boundary": boundary,
                    "start_pos": i,
                    "end_pos": i,
                    "positions": [i],
                })
            else:
                segs[-1]["end_pos"] = i
                segs[-1]["positions"].append(i)
        if rat == "NR":
            for seg in segs:
                g = None
                for j in range(seg["start_pos"], seg["end_pos"] + 1):
                    g = _nr_band_to_group(nr_band.iat[j])
                    if g is not None:
                        break
                seg["nr_band_group"] = g
        return segs

    def _assign(segs: List[dict], rat: str, out_series: pd.Series) -> int:
        cells_by_key: Dict[str, List[dict]] = {}
        for seg in segs:
            key = seg["key"]
            sp = seg["start_pos"]
            ep = seg["end_pos"]
            seg_lon = lon[sp]
            seg_lat = lat[sp]
            th_km = _lookup_cell_threshold_km(rat, seg.get("nr_band_group"))

            matched = None
            matched_dist = float("inf")
            matched_recent_end = -1
            for cell in cells_by_key.get(key, []):
                for pe in cell["seg_end_positions"]:
                    d = _haversine_km(lon[pe], lat[pe], seg_lon, seg_lat)
                    if math.isnan(d) or d > th_km:
                        continue
                    if d < matched_dist or (d == matched_dist and pe > matched_recent_end):
                        matched = cell
                        matched_dist = d
                        matched_recent_end = pe

            if matched is None:
                inst_idx = len(cells_by_key.get(key, []))
                matched = {"inst": inst_idx, "seg_end_positions": []}
                cells_by_key.setdefault(key, []).append(matched)

            matched["seg_end_positions"].append(ep)
            inst_key = f"{key}|inst={matched['inst']}"
            for p in seg["positions"]:
                out_series.iat[p] = inst_key

        # Count base keys that produced more than one instance.
        return sum(1 for v in cells_by_key.values() if len(v) > 1)

    nr_key = pd.Series(pd.NA, index=df.index, dtype="string")
    lte_key = pd.Series(pd.NA, index=df.index, dtype="string")
    nr_split = _assign(_build_segments(_nr_base_key, "NR"), "NR", nr_key)
    lte_split = _assign(_build_segments(_lte_base_key, "LTE"), "LTE", lte_key)
    if nr_split or lte_split:
        print(
            f"    cell-instance split: NR base keys with >1 instance = {nr_split}, "
            f"LTE = {lte_split}",
            flush=True,
        )
    return nr_key, lte_key


# ---------------------------------------------------------------------------
# Band / frequency helpers (from build_complete_ue_level_log.py)
# ---------------------------------------------------------------------------
def extract_band_number(band) -> Optional[int]:
    s = str(band).strip()
    m = re.search(r"\bBand\s*(\d+)\b", s, re.IGNORECASE) or re.search(r"\bb(\d+)\b", s, re.IGNORECASE)
    return int(m.group(1)) if m else None


LTE_BAND_TABLE = {
    2:  (1930.0,   600,   600,  1199),
    4:  (2110.0,  1950,  1950,  2399),
    5:  ( 869.0,  2400,  2400,  2649),
    12: ( 729.0,  5010,  5010,  5179),
    13: ( 746.0,  5180,  5180,  5279),
    14: ( 758.0,  5280,  5280,  5379),
    17: ( 734.0,  5730,  5730,  5849),
    25: (1930.0,  8040,  8040,  8689),
    26: ( 859.0,  8690,  8690,  9039),
    29: ( 717.0,  9660,  9660,  9769),
    30: (2350.0,  9770,  9770,  9869),
    41: (2496.0, 39650, 39650, 41589),
    66: (2110.0, 66436, 66436, 67335),
    71: ( 617.0, 68586, 68586, 68935),
}


def safe_lte_dl_freq_mhz(band, earfcn) -> Optional[float]:
    bn = extract_band_number(band)
    if bn is None or bn not in LTE_BAND_TABLE:
        return None
    try:
        e = int(float(str(earfcn)))
    except Exception:
        return None
    f_low, n_off, n_min, n_max = LTE_BAND_TABLE[bn]
    if not (n_min <= e <= n_max):
        return None
    return f_low + 0.1 * (e - n_off)


def nr_arfcn_to_freq_mhz(arfcn) -> Optional[float]:
    try:
        n = int(round(float(arfcn)))
    except Exception:
        return None
    if 0 <= n <= 599999:
        return 0.005 * n
    if 600000 <= n <= 2016666:
        return 3000.0 + 0.015 * (n - 600000)
    if 2016667 <= n <= 3279165:
        return 24250.08 + 0.06 * (n - 2016667)
    return None


def freq_to_key(freq_mhz) -> Optional[str]:
    try:
        f = float(freq_mhz)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return f"{f:.2f}".rstrip("0").rstrip(".")


def normalize_band_type(value) -> Optional[str]:
    s = clean_value(value)
    if s is None:
        return None
    s = s.lower()
    if s in {"low", "mid", "mmwave"}:
        return s
    if s in {"mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return None


def normalize_rat_label(rat) -> Optional[str]:
    s = str(rat).strip().upper()
    if s in {"NR", "5G"}:
        return "5G"
    if s in {"LTE", "4G"}:
        return "4G"
    return None


def load_band_lookups(band_summary_csv: Path) -> Tuple[Dict[Tuple[str, str], str],
                                                        Dict[Tuple[str, str], str],
                                                        Dict[Tuple[str, str], str]]:
    """Returns (by_band -> band_type, by_channel -> band_type, channel -> band)."""
    by_band: Dict[Tuple[str, str], str] = {}
    by_channel_type: Dict[Tuple[str, str], str] = {}
    channel_to_band: Dict[Tuple[str, str], str] = {}
    if not band_summary_csv.exists():
        return by_band, by_channel_type, channel_to_band
    df = pd.read_csv(band_summary_csv)
    required = {"rat", "band", "band_type", "channel_keys"}
    if not required.issubset(df.columns):
        return by_band, by_channel_type, channel_to_band
    for row in df.itertuples(index=False):
        rat = normalize_rat_label(getattr(row, "rat"))
        band = clean_value(getattr(row, "band"))
        band_type = normalize_band_type(getattr(row, "band_type"))
        if rat is None or band_type is None:
            continue
        if band is not None:
            by_band[(rat, band)] = band_type
        for ch in str(getattr(row, "channel_keys")).split(","):
            ch = clean_value(ch)
            if ch is not None:
                by_channel_type[(rat, ch)] = band_type
                if band is not None:
                    channel_to_band[(rat, ch)] = band
    return by_band, by_channel_type, channel_to_band


def infer_band_type(rat, freq_mhz) -> Optional[str]:
    try:
        f = float(freq_mhz)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    if f < 1000:
        return "low"
    if str(rat) == "5G" and f >= 24000:
        return "mmwave"
    return "mid"


# 3GPP DL freq ranges for US-relevant NR bands (MHz). Used as last-resort band fallback
# when same-operator lookup also fails. Order matters when ranges overlap — list more
# specific / common bands first so they win.
NR_BAND_FREQ_RANGES = [
    ("n71",   617.0,   652.0),
    ("n29",   717.0,   728.0),
    ("n12",   729.0,   746.0),
    ("n14",   758.0,   768.0),
    ("n5",    869.0,   894.0),
    ("n26",   859.0,   894.0),
    ("n2",   1930.0,  1995.0),
    ("n25",  1930.0,  1995.0),
    ("n66",  2110.0,  2200.0),
    ("n30",  2350.0,  2360.0),
    ("n41",  2496.0,  2690.0),  # n90 overlaps n41 (full-duplex variant); prefer n41
    ("n77",  3300.0,  4200.0),
    ("n78",  3300.0,  3800.0),
    ("n79",  4400.0,  5000.0),
    ("n261", 27500.0, 28350.0),
    ("n257", 26500.0, 29500.0),
    ("n258", 24250.0, 27500.0),
    ("n260", 37000.0, 40000.0),
    ("n262", 47200.0, 48200.0),
]

# LTE DL freq ranges (MHz) keyed to "Band N" labels.
LTE_BAND_FREQ_RANGES = [
    ("Band 71",  617.0,   652.0),
    ("Band 29",  717.0,   728.0),
    ("Band 12",  729.0,   746.0),
    ("Band 13",  746.0,   756.0),
    ("Band 14",  758.0,   768.0),
    ("Band 17",  734.0,   746.0),
    ("Band 5",   869.0,   894.0),
    ("Band 26",  859.0,   894.0),
    ("Band 2",  1930.0,  1990.0),
    ("Band 25", 1930.0,  1995.0),
    ("Band 4",  2110.0,  2155.0),
    ("Band 66", 2110.0,  2200.0),
    ("Band 30", 2350.0,  2360.0),
    ("Band 41", 2496.0,  2690.0),
]


def infer_band_from_freq(rat, freq_mhz) -> Optional[str]:
    try:
        f = float(freq_mhz)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    table = NR_BAND_FREQ_RANGES if str(rat) == "5G" else LTE_BAND_FREQ_RANGES
    for name, lo, hi in table:
        if lo <= f <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# Per-operator processing
# ---------------------------------------------------------------------------
def parse_neighbor_count(val) -> Optional[int]:
    s = clean_value(val)
    if s is None:
        return None
    parts = [p for p in (clean_value(x) for x in s.split(",")) if p is not None]
    return len(parts) if parts else 0


def build_per_cell_for_operator(in_csv: Path, operator: str,
                                 band_lookups) -> pd.DataFrame:
    print(f"  loading {in_csv} ...", flush=True)
    df = pd.read_csv(in_csv, low_memory=False)
    print(f"    rows={len(df):,}  cols={len(df.columns)}", flush=True)

    # 1) Discover components, add interference columns
    comps = discover_components(df)
    inter_cols = add_interference_columns(df, comps)
    print(f"    components: {[(c.rat, c.role, c.idx) for c in comps]}", flush=True)

    # 2) Drop rows with no usable PCell signal
    n_before = len(df)
    df = drop_rows_with_no_signal(df).reset_index(drop=True)
    print(f"    after drop_rows_with_no_signal: {n_before:,} -> {len(df):,}", flush=True)

    # 3) Build NR / LTE PCell cell keys
    nr_key, lte_key = build_cell_keys(df)
    df["nr_cell_key"] = nr_key
    df["lte_cell_key"] = lte_key

    # Forward-fill Event Technology (optional context column)
    if EVENT_TECH_COL in df.columns:
        df[EVENT_TECH_COL] = df[EVENT_TECH_COL].astype("string").replace({"": pd.NA}).ffill()

    by_band, by_channel_type, channel_to_band = band_lookups

    # 4) Per-RAT row split: emit a 5G row if NR PCell present, a 4G row if LTE PCell present
    nr_pci_num = to_num(df, NR_PCI_COL)
    lte_pci_num = to_num(df, LTE_PCI_COL)

    out_frames: List[pd.DataFrame] = []
    for rat, rat_label, pci_present_mask in [
        ("NR",  "5G", nr_pci_num.notna()),
        ("LTE", "4G", lte_pci_num.notna()),
    ]:
        if not pci_present_mask.any():
            continue
        sub = df.loc[pci_present_mask].copy()

        if rat == "NR":
            sub["cell_key"] = sub["nr_cell_key"]
            sub["pcell_pci"] = to_str(sub, NR_PCI_COL).map(clean_value)
            sub["pcell_arfcn"] = to_str(sub, NR_ARFCN_COL).map(clean_value)
            sub["pcell_earfcn"] = pd.NA
            sub["band"] = to_str(sub, NR_BAND_COL).map(clean_value)
            sub["rsrp"] = to_num(sub, NR_RSRP_COL)
            sub["sinr"] = to_num(sub, NR_SINR_COL)
            sub["interference"] = to_num(sub, inter_cols[("NR", "PCell", None)])
            sub["neighbor_list_raw"] = to_str(sub, NR_NEI_COL).map(clean_value)
            freq_from_col = to_num(sub, NR_FREQ_COL)
            sub["frequency_mhz"] = freq_from_col.where(
                freq_from_col.notna(),
                pd.Series([nr_arfcn_to_freq_mhz(v) for v in sub["pcell_arfcn"]],
                          index=sub.index, dtype="float64"),
            )
            sub["channel_key"] = pd.Series(
                [freq_to_key(v) for v in sub["frequency_mhz"]], index=sub.index, dtype="object"
            )
            # SCell aux columns (5G SCell components)
            for c in comps:
                if c.rat == "NR" and c.role == "SCell":
                    sub[f"scell{c.idx}_pci"] = to_num(sub, c.pci_col)
                    sub[f"scell{c.idx}_rsrp"] = to_num(sub, c.rsrp_col)
                    sub[f"scell{c.idx}_sinr"] = to_num(sub, c.sinr_col)
                    sub[f"scell{c.idx}_interference"] = to_num(sub, inter_cols[(c.rat, c.role, c.idx)])
        else:
            sub["cell_key"] = sub["lte_cell_key"]
            sub["pcell_pci"] = to_str(sub, LTE_PCI_COL).map(clean_value)
            sub["pcell_earfcn"] = to_str(sub, LTE_EARFCN_COL).map(clean_value)
            sub["pcell_arfcn"] = pd.NA
            sub["band"] = to_str(sub, LTE_BAND_COL).map(clean_value)
            sub["rsrp"] = to_num(sub, LTE_RSRP_COL)
            sub["sinr"] = to_num(sub, LTE_SINR_COL)
            sub["interference"] = to_num(sub, inter_cols[("LTE", "PCell", None)])
            sub["neighbor_list_raw"] = to_str(sub, LTE_NEI_COL).map(clean_value)
            sub["frequency_mhz"] = pd.Series(
                [safe_lte_dl_freq_mhz(b, e) for b, e in zip(sub["band"], sub["pcell_earfcn"])],
                index=sub.index, dtype="float64",
            )
            sub["channel_key"] = sub["pcell_earfcn"]
            for c in comps:
                if c.rat == "LTE" and c.role == "SCell":
                    sub[f"lte_scell{c.idx}_pci"] = to_num(sub, c.pci_col)
                    sub[f"lte_scell{c.idx}_rsrp"] = to_num(sub, c.rsrp_col)
                    sub[f"lte_scell{c.idx}_sinr"] = to_num(sub, c.sinr_col)
                    sub[f"lte_scell{c.idx}_interference"] = to_num(sub, inter_cols[(c.rat, c.role, c.idx)])

        sub["rat"] = rat_label
        sub["operator"] = operator
        sub["neighbor_count"] = sub["neighbor_list_raw"].map(parse_neighbor_count)

        # Band fill: per (operator, rat, cell_key), use the first non-null band seen
        first_band = (
            sub.dropna(subset=["band"])
               .drop_duplicates(subset=["cell_key"])
               .set_index("cell_key")["band"]
        )
        sub["band"] = sub["band"].fillna(sub["cell_key"].map(first_band))

        # Band fill via band_summary channel_to_band
        if channel_to_band:
            missing = sub["band"].isna() & sub["channel_key"].notna()
            if missing.any():
                sub.loc[missing, "band"] = [
                    channel_to_band.get((rat_label, str(ch)))
                    for ch in sub.loc[missing, "channel_key"]
                ]

        # Band fill #1: same operator + same frequency seen elsewhere with a band label.
        # Uses the most-common label per freq within this operator+rat.
        op_freq_band = (
            sub.dropna(subset=["band", "frequency_mhz"])
               .groupby("frequency_mhz")["band"]
               .agg(lambda s: s.value_counts().idxmax())
        )
        if not op_freq_band.empty:
            missing = sub["band"].isna() & sub["frequency_mhz"].notna()
            if missing.any():
                sub.loc[missing, "band"] = sub.loc[missing, "frequency_mhz"].map(op_freq_band)

        # Band fill #2: 3GPP standard freq ranges as last resort.
        missing = sub["band"].isna() & sub["frequency_mhz"].notna()
        if missing.any():
            sub.loc[missing, "band"] = [
                infer_band_from_freq(rat_label, f) for f in sub.loc[missing, "frequency_mhz"]
            ]

        # Consistency check: a row's freq must fall within its assigned band's 3GPP range.
        # If not (input data had a wrong band label that got forward-filled), override with
        # the 3GPP-inferred band for that frequency.
        ranges = NR_BAND_FREQ_RANGES if rat_label == "5G" else LTE_BAND_FREQ_RANGES
        band_range = {name: (lo, hi) for name, lo, hi in ranges}
        def _in_range(band, f):
            if band is None or pd.isna(band) or pd.isna(f):
                return True  # nothing to validate against
            lo_hi = band_range.get(str(band))
            if lo_hi is None:
                return True  # band not in our 3GPP table; trust input
            return lo_hi[0] <= float(f) <= lo_hi[1]
        bad = ~pd.Series([_in_range(b, f) for b, f in zip(sub["band"], sub["frequency_mhz"])], index=sub.index)
        if bad.any():
            sub.loc[bad, "band"] = [
                infer_band_from_freq(rat_label, f) for f in sub.loc[bad, "frequency_mhz"]
            ]

        # band_type lookup: by (rat, band) -> by (rat, channel) -> inferred from freq
        band_types = []
        for b, ch, f in zip(sub["band"], sub["channel_key"], sub["frequency_mhz"]):
            bt = by_band.get((rat_label, str(b))) if b is not None and not pd.isna(b) else None
            if bt is None and ch is not None and not pd.isna(ch):
                bt = by_channel_type.get((rat_label, str(ch)))
            if bt is None:
                bt = infer_band_type(rat_label, f)
            band_types.append(bt)
        sub["band_type"] = band_types

        # 2022 dataset note: T-Mobile mmWave is a single-site stationary test
        # (1 cell, ~30 min at one location), not real deployment coverage.
        # Mask as unknown so band_type-based plots ignore it.
        if operator == "tmobile" and rat_label == "5G":
            mmw = pd.Series(band_types, index=sub.index).eq("mmwave")
            if mmw.any():
                sub.loc[mmw, "band"] = None
                sub.loc[mmw, "band_type"] = None
                print(f"    masked {int(mmw.sum())} T-Mobile mmWave rows as unknown band", flush=True)

        out_frames.append(sub)

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)

    # 5) frequency_label per (operator, rat, band)
    out = add_operator_local_frequency_labels(out)

    # 6) Column ordering: put identity + derived cols up front, then everything else
    #    (we keep ALL input columns + all derived columns).
    leading = [
        "operator", "rat", "cell_key",
        TS_COL, LON_COL, LAT_COL,
        SRC_COL, RUN_COL,
        "band", "band_type", "frequency_mhz", "frequency_label",
        "channel_key", "pcell_pci", "pcell_arfcn", "pcell_earfcn",
        "rsrp", "sinr", "interference",
        "neighbor_count", "neighbor_list_raw",
    ]
    # SCell aux cols (NR + LTE)
    for c in comps:
        if c.role != "SCell":
            continue
        prefix = "scell" if c.rat == "NR" else "lte_scell"
        for suf in ("pci", "rsrp", "sinr", "interference"):
            name = f"{prefix}{c.idx}_{suf}"
            if name in out.columns and name not in leading:
                leading.append(name)

    leading = [c for c in leading if c in out.columns]
    rest = [c for c in out.columns if c not in leading]
    out = out[leading + rest].sort_values([TS_COL, "rat"]).reset_index(drop=True)
    return out


def add_operator_local_frequency_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 'frequency_label' like '<band>-f<idx>' per (operator, rat, band)."""
    if df.empty:
        df["frequency_label"] = pd.NA
        return df
    channels = (
        df[["operator", "rat", "band", "channel_key"]]
        .dropna(subset=["operator", "rat", "band", "channel_key"])
        .drop_duplicates()
        .copy()
    )
    channels["channel_value_num"] = pd.to_numeric(channels["channel_key"], errors="coerce")
    channels = channels.sort_values(
        ["operator", "rat", "band", "channel_value_num", "channel_key"],
        na_position="last",
    )
    channels["freq_idx"] = channels.groupby(["operator", "rat", "band"]).cumcount() + 1
    channels["frequency_label"] = channels.apply(
        lambda r: f"{r['band']}-f{int(r['freq_idx'])}", axis=1
    )
    label_map = {
        (str(r.operator), str(r.rat), str(r.band), str(r.channel_key)): str(r.frequency_label)
        for r in channels.itertuples(index=False)
    }
    df["frequency_label"] = [
        label_map.get((str(op), str(rat), str(b), str(ch)))
        if pd.notna(op) and pd.notna(rat) and pd.notna(b) and pd.notna(ch) else None
        for op, rat, b, ch in zip(df["operator"], df["rat"], df["band"], df["channel_key"])
    ]
    return df




def write_cell_counts(per_cell_csv: Path, output_csv: Path) -> None:
    """Count distinct observed cell instances for each operator and band."""
    seen = defaultdict(set)
    columns = ["operator", "rat", "band_type", "band", "cell_key"]
    for chunk in pd.read_csv(per_cell_csv, usecols=columns, chunksize=250_000):
        chunk = chunk.dropna(subset=columns)
        for key, part in chunk.groupby(["rat", "band_type", "band", "operator"], sort=False):
            seen[key].update(part["cell_key"].astype(str).unique())
    rows = [(*key, len(cells)) for key, cells in seen.items()]
    counts = pd.DataFrame(rows, columns=["rat", "band_type", "band", "operator", "cells"])
    wide = counts.pivot_table(
        index=["rat", "band_type", "band"], columns="operator",
        values="cells", fill_value=0, aggfunc="sum",
    ).reset_index()
    operators = ["att", "tmobile", "verizon"]
    for operator in operators:
        if operator not in wide.columns:
            wide[operator] = 0
        wide[operator] = wide[operator].astype(int)
    wide["total"] = wide[operators].sum(axis=1)
    wide = wide[["rat", "band_type", "band", *operators, "total"]]
    wide.sort_values(["rat", "band_type", "band"]).to_csv(output_csv, index=False)
    print(f"[OK] cell counts -> {output_csv}", flush=True)


def build_per_cell_outputs(in_dir: Path, out_dir: Path, band_summary: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    band_lookups = load_band_lookups(band_summary)

    operator_files = {
        "att":     in_dir / "AT&T.csv",
        "tmobile": in_dir / "T-Mobile.csv",
        "verizon": in_dir / "Verizon.csv",
    }

    all_frames: List[pd.DataFrame] = []
    for op, path in operator_files.items():
        if not path.exists():
            print(f"[WARN] missing {path}", flush=True)
            continue
        print(f"[INFO] processing {op} from {path.name}", flush=True)
        out_df = build_per_cell_for_operator(path, op, band_lookups)
        if out_df.empty:
            print(f"[WARN] {op}: no rows produced", flush=True)
            continue
        op_out = out_dir / f"{op}.csv"
        out_df.to_csv(op_out, index=False)
        print(f"[OK] {op}: rows={len(out_df):,}, cells={out_df['cell_key'].nunique():,} -> {op_out}", flush=True)
        all_frames.append(out_df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True).sort_values(
            ["operator", TS_COL, "rat"]
        ).reset_index(drop=True)
        all_out = out_dir / "per_cell_all.csv"
        combined.to_csv(all_out, index=False)
        print(f"[OK] combined: rows={len(combined):,} -> {all_out}", flush=True)
        del combined, all_frames
        write_cell_counts(all_out, out_dir / "per_operator_band_cell_counts.csv")


def load_device_map(path: Path) -> dict[str, str]:
    """Read the public alias-to-operator lookup used during run assignment."""
    mapping = pd.read_csv(path, dtype=str)
    required = {"device_id", "operator"}
    if not required.issubset(mapping.columns):
        raise ValueError(f"{path} needs columns: device_id, operator")
    result = {}
    for row in mapping.itertuples(index=False):
        device_id = normalize_phone(row.device_id)
        operator = str(row.operator).strip()
        if not device_id or operator not in {"AT&T", "T-Mobile", "Verizon"}:
            raise ValueError(f"Invalid device map entry in {path}")
        result[device_id] = operator
    if not result:
        raise ValueError(f"No devices in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcal-dir", type=Path,
                        default=REPO_ROOT / "data/raw_data/2022/xcal")
    parser.add_argument("--device-map", type=Path,
                        default=REPO_ROOT / "data/raw_data/2022/device_operator_map.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data/intermediate_data/2022")
    parser.add_argument("--band-summary", type=Path,
                        default=REPO_ROOT / "data/intermediate_data/2024/all_cells_band_summary.csv")
    parser.add_argument("--gap-s", type=float, default=2.0)
    args = parser.parse_args()

    if not args.xcal_dir.is_dir():
        parser.error(f"XCAL data directory does not exist: {args.xcal_dir}")
    if not args.device_map.is_file():
        parser.error(f"Device map does not exist: {args.device_map}")
    if not args.band_summary.is_file():
        parser.error(f"Band summary does not exist: {args.band_summary}")
    global PHONE_OP
    PHONE_OP = load_device_map(args.device_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".operator_split_", dir=args.out_dir) as work:
        operator_dir = Path(work)
        split_merged_by_operator(args.xcal_dir, operator_dir, args.gap_s)
        shutil.copy2(operator_dir / "operator_runs_log.csv",
                     args.out_dir / "operator_runs_log.csv")
        build_per_cell_outputs(operator_dir, args.out_dir, args.band_summary)


if __name__ == "__main__":
    main()
