#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# -----------------------------
# Filename parsing
# -----------------------------
FILENAME_RE = re.compile(r"^(?P<op>[A-Za-z])_(?P<day>\d+)_.*\.xlsx$", re.IGNORECASE)
OP_MAP = {"A": "att", "T": "tmobile", "V": "verizon"}

# -----------------------------
# Template column names
# -----------------------------
COL_UNIX_S = "Unix TIME_STAMP"  # epoch seconds (timezone-agnostic)
COL_TIME_STAMP = "TIME_STAMP"  # human time incl .mmm

COL_GCI = "Qualcomm Lte/LteAdv RRC Serving Cell Info Global Cell ID"
COL_NCI = "Qualcomm 5G-NR RRC Serving Cell Info NR Cell Identity[NCI]"

COL_EVENT_TECH = "Event Technology"  # optional / sparse in many exports

# -----------------------------
# KPI prefixes
# -----------------------------
NR_PREFIX = "5G KPI "
LTE_PREFIX = "LTE KPI "

ROLE_RE = re.compile(r"(?P<role>PCell|SCell\[(?P<idx>\d+)\])", re.IGNORECASE)


def parse_op_day(path: Path) -> Optional[Tuple[str, int]]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    op_letter = m.group("op").upper()
    day = int(m.group("day"))
    op = OP_MAP.get(op_letter)
    if not op:
        return None
    return op, day


def _as_clean_str_series(s: pd.Series) -> pd.Series:
    out = s.astype("string")
    out = out.str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out


def build_unix_ms(df: pd.DataFrame) -> pd.Series:
    """
    unix_ms = Unix TIME_STAMP * 1000 + milliseconds extracted from TIME_STAMP.
    If millis cannot be extracted, use 0.
    """
    if COL_UNIX_S not in df.columns:
        raise ValueError(f"Missing required column: {COL_UNIX_S}")

    unix_s = pd.to_numeric(df[COL_UNIX_S], errors="coerce")
    unix_ms = (unix_s * 1000.0).round().astype("Int64")

    ms_part = pd.Series([0] * len(df), index=df.index, dtype="Int64")
    if COL_TIME_STAMP in df.columns:
        ts = df[COL_TIME_STAMP].astype("string")
        m = ts.str.extract(r"\.(\d{1,3})\s*$", expand=False)  # digits after last '.'
        m = m.fillna("0").astype(str).str.zfill(3)
        ms_part = pd.to_numeric(m, errors="coerce").fillna(0).astype("Int64")

    return unix_ms + ms_part


def safe_event_tech_ffill(df: pd.DataFrame) -> None:
    if COL_EVENT_TECH in df.columns:
        s = df[COL_EVENT_TECH].astype("string")
        s = s.replace({"": pd.NA}).ffill()
        df[COL_EVENT_TECH] = s


_AREA_DF_CACHE: Optional[pd.DataFrame] = None
_THRESHOLD_MAP_CACHE: Optional[Dict[Tuple[str, str], float]] = None
_NR_BAND_GROUP_MAP = {
    "n2": "low",
    "n5": "low",
    "n12": "low",
    "n14": "low",
    "n25": "low",
    "n26": "low",
    "n66": "low",
    "n71": "low",
    "n41": "mid",
    "n48": "mid",
    "n77": "mid",
    "n78": "mid",
    "n79": "mid",
    "n260": "mmwave",
    "n261": "mmwave",
}


def _get_area_df_default() -> pd.DataFrame:
    global _AREA_DF_CACHE
    if _AREA_DF_CACHE is None:
        _AREA_DF_CACHE = pd.read_csv(ROOT / "data/raw_data/2024/area.csv")
    return _AREA_DF_CACHE


def _get_threshold_map_default() -> Dict[Tuple[str, str], float]:
    global _THRESHOLD_MAP_CACHE
    if _THRESHOLD_MAP_CACHE is None:
        tdf = pd.read_csv(ROOT / "data/raw_data/2024/threshold.csv")
        cols = {str(c).strip().lower(): c for c in tdf.columns}
        band_col = cols.get("band")
        type_col = cols.get("type")
        dist_col = cols.get("distance")
        if not (band_col and type_col and dist_col):
            raise ValueError("threshold.csv must contain Band, Type, Distance columns")
        out: Dict[Tuple[str, str], float] = {}
        for _, r in tdf.iterrows():
            band = str(r[band_col]).strip().lower()
            area = str(r[type_col]).strip().lower().replace("surburban", "suburban")
            dist = pd.to_numeric(pd.Series([r[dist_col]]), errors="coerce").iat[0]
            if band in {"", "nan"} or area in {"", "nan"} or pd.isna(dist):
                continue
            out[(band, area)] = float(dist)
        _THRESHOLD_MAP_CACHE = out
    return _THRESHOLD_MAP_CACHE


def _build_is_city_row_local(
    df: pd.DataFrame, area_df: pd.DataFrame, unix_ts_col: str = COL_UNIX_S
) -> pd.Series:
    out = pd.Series("unknown", index=df.index, dtype="object")
    if unix_ts_col not in df.columns:
        return out

    work = area_df.copy()
    work["utc_ts"] = pd.to_numeric(work["utc_ts"], errors="coerce")
    work["value"] = work["value"].astype(str).str.strip().str.lower()
    work["value"] = work["value"].replace({"surburban": "suburban"})
    work = work.dropna(subset=["utc_ts", "value"]).sort_values("utc_ts")
    work = work[work["value"].isin({"urban", "suburban", "rural"})]
    if work.empty:
        return out

    ts = pd.to_numeric(df[unix_ts_col], errors="coerce")
    valid_mask = ts.notna()
    if valid_mask.sum() == 0:
        return out

    left = pd.DataFrame(
        {"idx": df.index[valid_mask], "row_ts": ts[valid_mask].to_numpy(dtype=float)}
    ).sort_values("row_ts")
    right = work[["utc_ts", "value"]].rename(columns={"value": "area_type"})
    merged = pd.merge_asof(
        left, right, left_on="row_ts", right_on="utc_ts", direction="backward"
    )
    merged["area_type"] = merged["area_type"].fillna(right.iloc[0]["area_type"])
    out.loc[merged["idx"].to_numpy()] = merged["area_type"].to_numpy()
    return out


def _normalize_nr_band_group_local(x) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s == "" or s in {"nan", "none", "null"}:
        return None
    m = re.search(r"n\s*?(\d+)", s)
    if m:
        k = f"n{m.group(1)}"
        if k in _NR_BAND_GROUP_MAP:
            return _NR_BAND_GROUP_MAP[k]
    if s in {"low", "mid", "mmwave", "mm wave"}:
        return "mmwave" if s in {"mmwave", "mm wave"} else s
    return None


def _lookup_cell_threshold_km_local(
    threshold_map: Dict[Tuple[str, str], float],
    rat: str,
    area: str,
    nr_band_group: Optional[str],
    fallback_km: float = 20.0,
) -> float:
    area_key = str(area).strip().lower()
    if rat.upper() == "LTE":
        return float(threshold_map.get(("lte", area_key), fallback_km))
    if nr_band_group:
        return float(threshold_map.get((f"5g {nr_band_group}", area_key), fallback_km))
    return float(fallback_km)


def build_serving_id_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-row serving-id state based on PCell base keys.

    Updated logic:
      - Segment-level processing (contiguous runs of same base key per RAT, split on file boundary)
      - Segment-to-cell assignment uses area-aware threshold matching
      - Threshold comes from data/raw_data/2024/threshold.csv (LTE by area; NR by area + band-group)
      - Compare segment start against all prior segment-end points in candidate cells; first match wins

    Output schema is unchanged.
    """
    NR_PCI_COL = "5G KPI PCell RF Serving PCI"
    NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"
    NR_FREQ_COL = "5G KPI PCell RF Frequency [MHz]"
    NR_BAND_COL = "5G KPI PCell RF Band"

    LTE_PCI_COL = "LTE KPI PCell Serving PCI"
    LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"

    TS_COL = "Unix TIME_STAMP"
    LON_COL = "Lon"
    LAT_COL = "Lat"
    SRC_COL = "__source_file"

    CELL_MATCH_FALLBACK_KM = 20.0
    AREA_DOMINANCE_TH = 0.9
    GPS_JUMP_MILES = 0.06
    GPS_JUMP_KM = GPS_JUMP_MILES * 1.609344

    def _clean(x):
        if x is None or pd.isna(x):
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        if re.fullmatch(r"-?\d+\.0", s):
            s = s[:-2]
        return s

    def _to_num_series(col):
        if col not in df.columns:
            return pd.Series([pd.NA] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    def _to_str_series(col):
        if col not in df.columns:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
        return df[col].astype("string")

    def _haversine_km(lon1, lat1, lon2, lat2):
        if any(pd.isna(v) for v in (lon1, lat1, lon2, lat2)):
            return float("nan")
        rlon1, rlat1, rlon2, rlat2 = map(
            math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)]
        )
        dlon = rlon2 - rlon1
        dlat = rlat2 - rlat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return 6371.0 * c

    ts = _to_num_series(TS_COL)
    lon_raw = _to_num_series(LON_COL)
    lat_raw = _to_num_series(LAT_COL)

    # GPS jump filter: compare to previous raw row, hold previous output point if too large.
    lon_f = lon_raw.copy()
    lat_f = lat_raw.copy()
    prev_raw_lon = pd.NA
    prev_raw_lat = pd.NA
    prev_out_lon = pd.NA
    prev_out_lat = pd.NA
    for i in range(len(df)):
        lo = lon_raw.iat[i]
        la = lat_raw.iat[i]
        if (
            i == 0
            or pd.isna(prev_raw_lon)
            or pd.isna(prev_raw_lat)
            or pd.isna(lo)
            or pd.isna(la)
        ):
            lon_f.iat[i] = lo
            lat_f.iat[i] = la
            prev_raw_lon, prev_raw_lat = lo, la
            prev_out_lon, prev_out_lat = lon_f.iat[i], lat_f.iat[i]
            continue
        d_km = _haversine_km(prev_raw_lon, prev_raw_lat, lo, la)
        if (not math.isnan(d_km)) and (d_km > GPS_JUMP_KM):
            lon_f.iat[i] = prev_out_lon
            lat_f.iat[i] = prev_out_lat
        else:
            lon_f.iat[i] = lo
            lat_f.iat[i] = la
        prev_raw_lon, prev_raw_lat = lo, la
        prev_out_lon, prev_out_lat = lon_f.iat[i], lat_f.iat[i]

    nr_pci = _to_str_series(NR_PCI_COL).map(_clean)
    nr_arfcn = _to_str_series(NR_ARFCN_COL).map(_clean)
    nr_freq = _to_str_series(NR_FREQ_COL).map(_clean)
    nr_band = _to_str_series(NR_BAND_COL).map(_clean)

    lte_pci = _to_str_series(LTE_PCI_COL).map(_clean)
    lte_earfcn = _to_str_series(LTE_EARFCN_COL).map(_clean)
    src_file = _to_str_series(SRC_COL)

    try:
        area_df = _get_area_df_default()
        threshold_map = _get_threshold_map_default()
        row_area = (
            _build_is_city_row_local(df, area_df, unix_ts_col=TS_COL)
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"surburban": "suburban"})
        )
    except Exception:
        row_area = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
        threshold_map = {}

    def _nr_base_key(i):
        if nr_pci.iat[i] is None:
            return None
        return f"NR|PCell|pci={nr_pci.iat[i]}|arfcn={nr_arfcn.iat[i]}|freq={nr_freq.iat[i]}"

    def _lte_base_key(i):
        if lte_pci.iat[i] is None:
            return None
        return f"LTE|PCell|pci={lte_pci.iat[i]}|earfcn={lte_earfcn.iat[i]}"

    def _segment_area_type(start_pos: int, end_pos: int) -> str:
        areas = (
            row_area.iloc[start_pos : end_pos + 1]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"surburban": "suburban"})
        )
        ok = areas.isin(["urban", "suburban", "rural"])
        if ok.sum() == 0:
            return "unknown"
        counts = areas[ok].value_counts()
        top_area = str(counts.index[0])
        frac = float(counts.iloc[0] / counts.sum())
        return top_area if frac >= AREA_DOMINANCE_TH else "unknown"

    def _segment_nr_band_group(start_pos: int, end_pos: int) -> Optional[str]:
        for j in range(start_pos, end_pos + 1):
            g = _normalize_nr_band_group_local(nr_band.iat[j])
            if g is not None:
                return g
        return None

    def _build_segments(base_key_fn, rat: str) -> List[dict]:
        key_cache: List[Optional[str]] = [None] * len(df)
        valid_pos: List[int] = []
        for i in range(len(df)):
            bk = base_key_fn(i)
            key_cache[i] = bk
            if bk is not None:
                valid_pos.append(i)

        segs: List[dict] = []
        for pos in valid_pos:
            bk = str(key_cache[pos])
            f = str(src_file.iat[pos]) if pd.notna(src_file.iat[pos]) else ""
            if (not segs) or segs[-1]["key"] != bk or segs[-1]["file_name"] != f:
                segs.append(
                    {
                        "key": bk,
                        "file_name": f,
                        "start_pos": pos,
                        "end_pos": pos,
                        "positions": [pos],
                    }
                )
            else:
                segs[-1]["end_pos"] = pos
                segs[-1]["positions"].append(pos)

        for seg in segs:
            seg["area"] = _segment_area_type(seg["start_pos"], seg["end_pos"])
            seg["nr_band_group"] = (
                _segment_nr_band_group(seg["start_pos"], seg["end_pos"])
                if rat == "NR"
                else None
            )
        return segs

    def _assign_segments(
        segs: List[dict], rat: str, out_series: pd.Series, out_area_series: pd.Series
    ) -> None:
        cells_by_key: Dict[str, List[dict]] = {}
        for seg in segs:
            if seg["area"] == "unknown":
                continue

            key = seg["key"]
            seg_start = seg["start_pos"]
            seg_end = seg["end_pos"]
            seg_lon = lon_f.iat[seg_start]
            seg_lat = lat_f.iat[seg_start]
            seg_ts = ts.iat[seg_start]

            matched_cell = None
            matched_dist = float("inf")
            matched_recent_end = -1
            for cell in cells_by_key.get(key, []):
                th_km = cell["th_km"]
                for prev_end_pos in cell["seg_end_positions"]:
                    d_km = _haversine_km(
                        lon_f.iat[prev_end_pos],
                        lat_f.iat[prev_end_pos],
                        seg_lon,
                        seg_lat,
                    )
                    if math.isnan(d_km) or d_km > th_km:
                        continue
                    # Best-match: smallest feasible distance.
                    # Tie-break by more recent segment end to keep assignment deterministic.
                    if d_km < matched_dist or (
                        d_km == matched_dist and prev_end_pos > matched_recent_end
                    ):
                        matched_cell = cell
                        matched_dist = d_km
                        matched_recent_end = prev_end_pos

            if matched_cell is None:
                t0 = int(seg_ts) if pd.notna(seg_ts) else seg_start
                th_km = _lookup_cell_threshold_km_local(
                    threshold_map,
                    rat=rat,
                    area=seg["area"],
                    nr_band_group=seg.get("nr_band_group"),
                    fallback_km=CELL_MATCH_FALLBACK_KM,
                )
                matched_cell = {
                    "t0": t0,
                    "area": seg["area"],
                    "nr_band_group": seg.get("nr_band_group"),
                    "th_km": th_km,
                    "seg_end_positions": [],
                }
                cells_by_key.setdefault(key, []).append(matched_cell)

            matched_cell["seg_end_positions"].append(seg_end)
            inst_key = f"{key}|t0={matched_cell['t0']}"
            for pos in seg["positions"]:
                out_series.iat[pos] = inst_key
                out_area_series.iat[pos] = matched_cell["area"]

    pcell_key_nr = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    pcell_key_lte = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    pcell_area_nr = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    pcell_area_lte = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    _assign_segments(
        _build_segments(_nr_base_key, "NR"), "NR", pcell_key_nr, pcell_area_nr
    )
    _assign_segments(
        _build_segments(_lte_base_key, "LTE"), "LTE", pcell_key_lte, pcell_area_lte
    )

    serving_type = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    serving_val = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    for i, idx in enumerate(df.index):
        nrk = pcell_key_nr.at[idx]
        ltek = pcell_key_lte.at[idx]
        if pd.notna(nrk):
            serving_type.at[idx] = "NR_PCell"
            serving_val.at[idx] = nrk
        elif pd.notna(ltek):
            serving_type.at[idx] = "LTE_PCell"
            serving_val.at[idx] = ltek

    df["pcell_key_nr"] = pcell_key_nr
    df["pcell_key_lte"] = pcell_key_lte
    df["__pcell_area_assigned_nr"] = pcell_area_nr
    df["__pcell_area_assigned_lte"] = pcell_area_lte
    df["serving_id_type"] = serving_type
    df["serving_id_value"] = serving_val
    df["serving_id"] = (
        serving_type.astype("string") + ":" + serving_val.astype("string")
    )
    return df


def build_pcell_area_type_map(
    df: pd.DataFrame,
    pcell_index: Dict[str, np.ndarray],
    dominance_threshold: float = 0.9,
) -> Dict[str, str]:
    """
    Compute dominant area type per pcell key using row-level area labels.

    Returns key -> area_type where area_type in {"urban","suburban","rural","unknown"}.
    """
    # Prefer the exact assignment-time cell area used in build_serving_id_state.
    assigned_nr = (
        df["__pcell_area_assigned_nr"].astype("string")
        if "__pcell_area_assigned_nr" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )
    assigned_lte = (
        df["__pcell_area_assigned_lte"].astype("string")
        if "__pcell_area_assigned_lte" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )

    if (
        "__pcell_area_assigned_nr" in df.columns
        or "__pcell_area_assigned_lte" in df.columns
    ):
        out_assigned: Dict[str, str] = {}
        for key, idxs in pcell_index.items():
            rat = str(key).split("|", 1)[0].upper() if isinstance(key, str) else ""
            src = assigned_nr if rat == "NR" else assigned_lte if rat == "LTE" else None
            if src is None:
                out_assigned[key] = "unknown"
                continue
            vals = (
                src.loc[idxs]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"surburban": "suburban", "<na>": "unknown"})
            )
            vals = vals[vals.isin(["urban", "suburban", "rural"])]
            out_assigned[key] = str(vals.iloc[0]) if len(vals) else "unknown"
        return out_assigned

    try:
        area_df = _get_area_df_default()
        row_area = (
            _build_is_city_row_local(df, area_df, unix_ts_col=COL_UNIX_S)
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"surburban": "suburban"})
        )
    except Exception:
        return {k: "unknown" for k in pcell_index.keys()}

    out: Dict[str, str] = {}
    for key, idxs in pcell_index.items():
        areas = row_area.loc[idxs]
        area_ok = areas.isin(["urban", "suburban", "rural"])
        if area_ok.sum() == 0:
            out[key] = "unknown"
            continue
        counts = areas[area_ok].value_counts()
        top_area = str(counts.index[0])
        frac = float(counts.iloc[0] / counts.sum())
        out[key] = top_area if frac >= dominance_threshold else "unknown"
    return out


@dataclass(frozen=True)
class ComponentCols:
    rat: str  # "NR" or "LTE"
    role: str  # "PCell" or "SCell"
    idx: Optional[int]  # SCell index, else None
    pci_col: str
    rsrp_col: str
    sinr_col: str


def discover_components(df: pd.DataFrame) -> List[ComponentCols]:
    """
    Dynamically find all PCell/SCell components that have PCI+RSRP+SINR columns.

    Requirement:
      - Number of SCells varies across files.
      - If a given SCell lacks rsrp or sinr column, ditch that SCell.
    """
    temp: Dict[Tuple[str, str, Optional[int]], Dict[str, str]] = {}

    for col in df.columns:
        col_str = str(col)

        if col_str.startswith(NR_PREFIX):
            rat = "NR"
            body = col_str[len(NR_PREFIX) :]
        elif col_str.startswith(LTE_PREFIX):
            rat = "LTE"
            body = col_str[len(LTE_PREFIX) :]
        else:
            continue

        rm = ROLE_RE.search(body)
        if not rm:
            continue

        role_raw = rm.group("role")
        idx = rm.group("idx")
        if role_raw.lower().startswith("pcell"):
            role = "PCell"
            sc_idx = None
        else:
            role = "SCell"
            sc_idx = int(idx) if idx is not None else None

        kind: Optional[str] = None
        if re.search(r"\bPCI\b", col_str, re.IGNORECASE):
            kind = "pci"
        elif re.search(r"RSRP", col_str, re.IGNORECASE):
            kind = "rsrp"
        elif re.search(r"SINR", col_str, re.IGNORECASE):
            kind = "sinr"

        if kind is None:
            continue

        key = (rat, role, sc_idx)
        temp.setdefault(key, {})[kind] = col_str

    comps: List[ComponentCols] = []
    for (rat, role, sc_idx), d in temp.items():
        if {"pci", "rsrp", "sinr"}.issubset(d.keys()):
            comps.append(
                ComponentCols(
                    rat=rat,
                    role=role,
                    idx=sc_idx,
                    pci_col=d["pci"],
                    rsrp_col=d["rsrp"],
                    sinr_col=d["sinr"],
                )
            )

    # Stable ordering
    def _sort_key(c: ComponentCols) -> Tuple[int, int, int]:
        rat_ord = 0 if c.rat == "NR" else 1
        role_ord = 0 if c.role == "PCell" else 1
        idx_ord = c.idx if c.idx is not None else -1
        return (rat_ord, role_ord, idx_ord)

    comps.sort(key=_sort_key)
    return comps


def add_interference_columns(
    df: pd.DataFrame,
    comps: List[ComponentCols],
) -> Dict[Tuple[str, str, Optional[int]], str]:
    """
    Add interference columns for each analyzable component:
      Interference(dBm) = RSRP(dBm) - SINR(dB) only when both are present, else NaN.
    Returns mapping (rat, role, idx)-> new_column_name.
    """
    out: Dict[Tuple[str, str, Optional[int]], str] = {}
    for c in comps:
        if c.role == "PCell":
            name = f"{c.rat} KPI PCell Interference [dBm]"
        else:
            name = f"{c.rat} KPI SCell[{c.idx}] Interference [dBm]"

        rsrp = pd.to_numeric(df[c.rsrp_col], errors="coerce")
        sinr = pd.to_numeric(df[c.sinr_col], errors="coerce")
        inter = (rsrp - sinr).where(rsrp.notna() & sinr.notna(), other=np.nan)
        df[name] = inter
        out[(c.rat, c.role, c.idx)] = name
    return out


def drop_rows_with_no_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row filter based on the *PCell* KPIs for NR and LTE.

    Rule:
      1) If BOTH NR PCI and LTE PCI are empty => drop row.
      2) If only one RAT has PCI:
           - drop row if that RAT is missing either (RSRP or SINR/RSRQ).
      3) If BOTH RATs have PCI:
           - keep row if at least ONE RAT has both (RSRP and SINR/RSRQ)
           - drop row only if BOTH RATs lack complete (RSRP and SINR/RSRQ).

    Column names (as in your exports):
      - NR:  "5G KPI PCell RF Serving PCI"
             "5G KPI PCell RF Serving SS-RSRP [dBm]"
             "5G KPI PCell RF Serving SS-SINR [dB]"
      - LTE: "LTE KPI PCell Serving PCI"
             "LTE KPI PCell Serving RSRP [dBm]"
             "LTE KPI PCell Serving RSRQ [dB]"
    """
    NR_PCI = "5G KPI PCell RF Serving PCI"
    NR_RSRP = "5G KPI PCell RF Serving SS-RSRP [dBm]"
    NR_SINR = "5G KPI PCell RF Serving SS-SINR [dB]"

    LTE_PCI = "LTE KPI PCell Serving PCI"
    LTE_RSRP = "LTE KPI PCell Serving RSRP [dBm]"
    LTE_RSRQ = "LTE KPI PCell Serving RSRQ [dB]"

    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    nr_pci = _num(NR_PCI)
    lte_pci = _num(LTE_PCI)

    nr_rsrp = _num(NR_RSRP)
    nr_sinr = _num(NR_SINR)

    lte_rsrp = _num(LTE_RSRP)
    lte_rsrq = _num(LTE_RSRQ)

    nr_present = nr_pci.notna()
    lte_present = lte_pci.notna()

    nr_complete = nr_present & nr_rsrp.notna() & nr_sinr.notna()
    lte_complete = lte_present & lte_rsrp.notna() & lte_rsrq.notna()

    keep_only_nr = nr_present & (~lte_present) & nr_complete
    keep_only_lte = (~nr_present) & lte_present & lte_complete
    keep_both = nr_present & lte_present & (nr_complete | lte_complete)

    keep = keep_only_nr | keep_only_lte | keep_both
    return df.loc[keep].copy()


def build_pcell_key_series(
    df: pd.DataFrame, comps: List[ComponentCols]
) -> Dict[str, pd.Series]:
    """
    Build per-row PCell keys for NR and LTE.

    Preferred (new) behavior:
      - If df already has 'pcell_key_nr' / 'pcell_key_lte' (from build_serving_id_state),
        return them directly.

    Fallback (legacy behavior):
      - "RAT|PCell|pci=<pci>|sid=<serving_id>"
    """
    keys: Dict[str, pd.Series] = {}

    if "pcell_key_nr" in df.columns:
        keys["pcell_key_nr"] = df["pcell_key_nr"].astype("string")
    if "pcell_key_lte" in df.columns:
        keys["pcell_key_lte"] = df["pcell_key_lte"].astype("string")

    if keys:
        return keys

    # ---- legacy fallback ----
    for rat in ["NR", "LTE"]:
        pc = next((c for c in comps if c.rat == rat and c.role == "PCell"), None)
        if pc is None:
            continue

        pci = pd.to_numeric(df[pc.pci_col], errors="coerce")
        sid = (
            df["serving_id"].astype("string")
            if "serving_id" in df.columns
            else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
        )

        key = rat + "|PCell|pci=" + pci.astype("Int64").astype("string") + "|sid=" + sid
        key = key.where(pci.notna() & sid.notna(), other=pd.NA)
        keys[f"pcell_key_{rat.lower()}"] = key

    return keys


def build_cell_indices(
    df: pd.DataFrame,
    comps: List[ComponentCols],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build:
      pcell_index: key -> row indices (np.ndarray of int positions in df)
      scell_index: key (includes attaching pcell_key) -> row indices

    IMPORTANT: values are indices so you can slice df and keep *all columns*.
    """
    pcell_keys = build_pcell_key_series(df, comps)
    pcell_index: Dict[str, np.ndarray] = {}

    # PCell indices for both RATs (if present)
    for rat in ["NR", "LTE"]:
        kcol = f"pcell_key_{rat.lower()}"
        if kcol not in pcell_keys:
            continue
        key_series = pcell_keys[kcol]
        # groupby on key values using numpy for speed
        valid = key_series.notna()
        if not valid.any():
            continue
        keys_arr = key_series[valid].astype("string").to_numpy()
        idx_arr = df.index.to_numpy()[valid.to_numpy()]
        # stable grouping via dict of lists
        tmp: Dict[str, List[int]] = {}
        for k, i in zip(keys_arr, idx_arr):
            tmp.setdefault(str(k), []).append(int(i))
        for k, lst in tmp.items():
            pcell_index[k] = np.array(lst, dtype=np.int64)

    # SCell indices: key includes attaching PCell key at that row
    scell_index: Dict[str, np.ndarray] = {}

    anchor_nr = pcell_keys.get(
        "pcell_key_nr", pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )
    anchor_lte = pcell_keys.get(
        "pcell_key_lte", pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )

    for c in comps:
        if c.role != "SCell":
            continue
        anchor = anchor_nr if c.rat == "NR" else anchor_lte
        if anchor.isna().all():
            continue

        pci = pd.to_numeric(df[c.pci_col], errors="coerce")
        sc_key = (
            anchor.astype("string")
            + "|"
            + c.rat
            + "|SCell["
            + str(c.idx)
            + "]|pci="
            + pci.astype("Int64").astype("string")
        )
        sc_key = sc_key.where(anchor.notna() & pci.notna(), other=pd.NA)

        valid = sc_key.notna()
        if not valid.any():
            continue
        keys_arr = sc_key[valid].astype("string").to_numpy()
        idx_arr = df.index.to_numpy()[valid.to_numpy()]

        tmp: Dict[str, List[int]] = {}
        for k, i in zip(keys_arr, idx_arr):
            tmp.setdefault(str(k), []).append(int(i))
        for k, lst in tmp.items():
            scell_index[k] = np.array(lst, dtype=np.int64)

    return pcell_index, scell_index


def add_run_id(df: pd.DataFrame, gap_ms: int = 5000) -> None:
    """
    Add df['run_id'] by splitting whenever time gap > gap_ms and/or __source_file changes.

    - gap_ms: threshold to split runs (e.g., 5000 ms works well for stopping/starting tests)
    """
    if "unix_ms" not in df.columns:
        raise ValueError("unix_ms must exist before add_run_id()")

    t = pd.to_numeric(df["unix_ms"], errors="coerce")
    dt = t.diff()

    gap_break = dt.isna() | (dt > gap_ms)

    # run_id starts at 1 for first run; Int64 keeps NA-compatible dtype
    df["run_id"] = gap_break.cumsum().astype("Int64")


def build_intra_freq_handover_events(
    df: pd.DataFrame,
    comps: List[Any],  # ComponentCols list; using Any to avoid re-imports here
    inter_cols: Dict[Tuple[str, str, Optional[int]], str],
    max_adj_gap_ms: int = 500,  # filter "adjacent samples" jumps (safety net)
) -> pd.DataFrame:
    """
    Intra-frequency PCell HO detection, run-aware:

    For each RAT (NR/LTE) and each run_id:
      - consider rows where PCell PCI and RF (ARFCN/EARFCN) are present
      - HO event when consecutive valid samples have:
          * same RF
          * different PCI
          * time gap <= max_adj_gap_ms
      - record before/after: PCI, RF, serving_id, rsrp/sinr/interference, timestamps, run_id

    Requires df['run_id'] be present (call add_run_id first).
    """
    events: List[Dict[str, Any]] = []

    if "run_id" not in df.columns:
        raise ValueError(
            "run_id missing. Call add_run_id(df, ...) before building HO events."
        )

    # Your chosen RF columns
    NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"
    LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"

    for rat in ["NR", "LTE"]:
        # Find PCell component discovered from headers
        pc = next((c for c in comps if c.rat == rat and c.role == "PCell"), None)
        if pc is None:
            continue

        rf_col = NR_ARFCN_COL if rat == "NR" else LTE_EARFCN_COL
        if rf_col not in df.columns:
            continue

        pci = pd.to_numeric(df[pc.pci_col], errors="coerce")
        rf = pd.to_numeric(df[rf_col], errors="coerce")

        rsrp = pd.to_numeric(df[pc.rsrp_col], errors="coerce")
        sinr = pd.to_numeric(df[pc.sinr_col], errors="coerce")

        inter_name = inter_cols.get((rat, "PCell", None))
        if inter_name and inter_name in df.columns:
            inter = pd.to_numeric(df[inter_name], errors="coerce")
        else:
            inter = pd.Series(np.nan, index=df.index)

        unix_ms = pd.to_numeric(df["unix_ms"], errors="coerce")
        serving_id = (
            df["serving_id"].astype("string")
            if "serving_id" in df.columns
            else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
        )
        run_id = df["run_id"]

        # Process each run separately
        for rid, run_idx in df.groupby("run_id", sort=False).groups.items():
            # run_idx is an Index of row labels
            run_idx = pd.Index(run_idx)

            # valid rows in this run: pci and rf and unix_ms present
            v = (
                pci.loc[run_idx].notna()
                & rf.loc[run_idx].notna()
                & unix_ms.loc[run_idx].notna()
            )
            if v.sum() < 2:
                continue

            idx = run_idx.to_numpy()[v.to_numpy()]
            # Ensure increasing time order (just in case)
            idx = idx[np.argsort(unix_ms.loc[idx].to_numpy())]

            pre_idx = idx[:-1]
            post_idx = idx[1:]

            pci_pre = pci.loc[pre_idx].to_numpy()
            pci_post = pci.loc[post_idx].to_numpy()
            rf_pre = rf.loc[pre_idx].to_numpy()
            rf_post = rf.loc[post_idx].to_numpy()

            t_pre = unix_ms.loc[pre_idx].to_numpy()
            t_post = unix_ms.loc[post_idx].to_numpy()
            gap = t_post - t_pre

            mask = (rf_pre == rf_post) & (pci_pre != pci_post) & (gap <= max_adj_gap_ms)
            if not np.any(mask):
                continue

            pre_idx = pre_idx[mask]
            post_idx = post_idx[mask]

            for i_pre, i_post in zip(pre_idx, post_idx):
                events.append(
                    {
                        "run_id": int(rid) if pd.notna(rid) else np.nan,
                        "rat": rat,
                        "unix_ms_pre": (
                            int(unix_ms.loc[i_pre])
                            if pd.notna(unix_ms.loc[i_pre])
                            else np.nan
                        ),
                        "unix_ms_post": (
                            int(unix_ms.loc[i_post])
                            if pd.notna(unix_ms.loc[i_post])
                            else np.nan
                        ),
                        "gap_ms": (
                            int(unix_ms.loc[i_post] - unix_ms.loc[i_pre])
                            if (
                                pd.notna(unix_ms.loc[i_pre])
                                and pd.notna(unix_ms.loc[i_post])
                            )
                            else np.nan
                        ),
                        "serving_id_pre": serving_id.loc[i_pre],
                        "serving_id_post": serving_id.loc[i_post],
                        "pcell_pci_pre": (
                            float(pci.loc[i_pre])
                            if pd.notna(pci.loc[i_pre])
                            else np.nan
                        ),
                        "pcell_pci_post": (
                            float(pci.loc[i_post])
                            if pd.notna(pci.loc[i_post])
                            else np.nan
                        ),
                        "rf_col": rf_col,
                        "rf_pre": (
                            float(rf.loc[i_pre]) if pd.notna(rf.loc[i_pre]) else np.nan
                        ),
                        "rf_post": (
                            float(rf.loc[i_post])
                            if pd.notna(rf.loc[i_post])
                            else np.nan
                        ),
                        "rsrp_pre": (
                            float(rsrp.loc[i_pre])
                            if pd.notna(rsrp.loc[i_pre])
                            else np.nan
                        ),
                        "rsrp_post": (
                            float(rsrp.loc[i_post])
                            if pd.notna(rsrp.loc[i_post])
                            else np.nan
                        ),
                        "sinr_pre": (
                            float(sinr.loc[i_pre])
                            if pd.notna(sinr.loc[i_pre])
                            else np.nan
                        ),
                        "sinr_post": (
                            float(sinr.loc[i_post])
                            if pd.notna(sinr.loc[i_post])
                            else np.nan
                        ),
                        "interference_pre": (
                            float(inter.loc[i_pre])
                            if pd.notna(inter.loc[i_pre])
                            else np.nan
                        ),
                        "interference_post": (
                            float(inter.loc[i_post])
                            if pd.notna(inter.loc[i_post])
                            else np.nan
                        ),
                    }
                )

    ho_df = pd.DataFrame(events)
    if not ho_df.empty:
        ho_df["delta_interference"] = (
            ho_df["interference_post"] - ho_df["interference_pre"]
        )
        ho_df["delta_rsrp"] = ho_df["rsrp_post"] - ho_df["rsrp_pre"]
        ho_df["delta_sinr"] = ho_df["sinr_post"] - ho_df["sinr_pre"]
    return ho_df


def load_and_concat(files: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        df = df.iloc[:-8]  # drop last 8 rows of summary stats
        df["__source_file"] = f.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def process_group(op: str, files: List[Path], out_dir: Path) -> Tuple[Path, Path, Path]:
    df = load_and_concat(files)
    if df.empty:
        raise RuntimeError(f"No data for {op}")
    day = 0  # combined-across-days output; keep numeric for downstream filename regex compatibility

    # Additive derived columns
    df["unix_ms"] = build_unix_ms(df)
    safe_event_tech_ffill(df)
    df = build_serving_id_state(df)

    # Drop completely-empty signal rows (your updated requirement)
    df = drop_rows_with_no_signal(df)

    # Discover analyzable components (dynamic SCell counts; ditch SCells without RSRP/SINR)
    comps = discover_components(df)

    # Add interference columns (still keeps all other columns)
    inter_cols = add_interference_columns(df, comps)

    # Build cell-perspective indices (key -> row indices)
    pcell_index, scell_index = build_cell_indices(df, comps)

    # Drop PCells with insufficient valid PCell interference samples.
    MIN_VALID_INTERFERENCE_ROWS_PER_PCELL = 30
    rat_pcell_inter_col: Dict[str, str] = {}
    for c in comps:
        if c.role == "PCell" and c.idx is None:
            name = inter_cols.get((c.rat, c.role, c.idx))
            if name and name in df.columns:
                rat_pcell_inter_col[c.rat] = name

    drop_pcells = set()
    for k, v in pcell_index.items():
        rat = str(k).split("|", 1)[0].upper() if isinstance(k, str) else ""
        inter_col = rat_pcell_inter_col.get(rat)
        if not inter_col:
            drop_pcells.add(k)
            continue
        valid_n = pd.to_numeric(df.loc[v, inter_col], errors="coerce").notna().sum()
        if int(valid_n) < MIN_VALID_INTERFERENCE_ROWS_PER_PCELL:
            drop_pcells.add(k)
    if drop_pcells:
        for k in drop_pcells:
            pcell_index.pop(k, None)
        # scell keys are formatted like "...|pcell=<pcell_key>" in build_cell_indices(); drop those whose pcell is dropped.
        new_scell_index = {}
        for sk, sv in scell_index.items():
            keep = True
            try:
                # look for "|pcell=<...>" suffix
                if "|pcell=" in sk:
                    pk = sk.split("|pcell=", 1)[1]
                    if pk in drop_pcells:
                        keep = False
            except Exception:
                pass
            if keep:
                new_scell_index[sk] = sv
        scell_index = new_scell_index  # Build intra-freq handover table (PCell only)

    pcell_area_type = build_pcell_area_type_map(
        df, pcell_index, dominance_threshold=0.9
    )
    add_run_id(df, gap_ms=5000)
    ho_df = build_intra_freq_handover_events(df, comps, inter_cols, max_adj_gap_ms=500)

    # Write pickles
    out_dir.mkdir(parents=True, exist_ok=True)
    pcell_pkl = out_dir / f"{op}_pcell.pkl"
    scell_pkl = out_dir / f"{op}_scell.pkl"
    ho_pkl = out_dir / f"{op}_pcell_ho.pkl"

    pcell_payload = {
        "meta": {
            "operator": op,
            "day": day,
            "days_combined": True,
            "files": [f.name for f in files],
            "n_rows_cleaned": int(len(df)),
            "n_components_analyzable": int(len(comps)),
            "pcell_area_dominance_threshold": 0.9,
            "pcell_area_type": pcell_area_type,
            "note": "df keeps almost all original columns + derived columns; cell perspective via indices.",
        },
        "components": [
            {
                "rat": c.rat,
                "role": c.role,
                "idx": c.idx,
                "pci_col": c.pci_col,
                "rsrp_col": c.rsrp_col,
                "sinr_col": c.sinr_col,
                "interference_col": inter_cols[(c.rat, c.role, c.idx)],
            }
            for c in comps
        ],
        "df": df,  # contains Lon/Lat/ARFCN/Band/BW/Freq/Throughput/etc as exported
        "pcell_index": pcell_index,  # key -> np.ndarray indices into df
    }
    pd.to_pickle(pcell_payload, pcell_pkl)

    scell_payload = {
        "meta": {
            "operator": op,
            "day": day,
            "days_combined": True,
            "files": [f.name for f in files],
            "n_scells": int(len(scell_index)),
            "note": "scell_index keys include attaching pcell_key; slice df from pcell pickle to get full columns.",
        },
        "pcell_pickle": pcell_pkl.name,
        "scell_index": scell_index,
    }
    pd.to_pickle(scell_payload, scell_pkl)

    ho_payload = {
        "meta": {
            "operator": op,
            "day": day,
            "days_combined": True,
            "files": [f.name for f in files],
            "definition": "Intra-freq HO: consecutive-row PCell key change with equal PCell RF context (best-effort).",
        },
        "pcell_pickle": pcell_pkl.name,
        "handover_df": ho_df,
    }
    pd.to_pickle(ho_payload, ho_pkl)

    return pcell_pkl, scell_pkl, ho_pkl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse XCAL Excel logs into per-operator cell-perspective pickles (all matched files combined)."
    )
    ap.add_argument(
        "--input-dir",
        type=str,
        # required=True,
        help="Directory containing *.xlsx like A_1_100MS.xlsx",
        default=("data/raw_data/2024/xcal"),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        # required=True,
        help="Directory to write pickles",
        default=("data/intermediate_data/2024/pkl_output"),
    )
    ap.add_argument(
        "--glob", type=str, default="*.xlsx", help="Glob pattern (default: *.xlsx)"
    )
    args = ap.parse_args()

    curr_dir = ROOT
    in_dir: Path = curr_dir / args.input_dir
    out_dir: Path = curr_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    groups: Dict[str, List[Path]] = {}
    for f in files:
        od = parse_op_day(f)
        if od is None:
            continue
        op, _day = od
        groups.setdefault(op, []).append(f)

    if not groups:
        raise SystemExit(f"No matching files found in {in_dir} (pattern {args.glob}).")

    for op, fs in sorted(groups.items(), key=lambda x: x[0]):
        pcell_pkl, scell_pkl, ho_pkl = process_group(op, fs, out_dir)
        print(f"[OK] {op} (all matched files combined):")
        print(f"  - {pcell_pkl.name}")
        print(f"  - {scell_pkl.name}")
        print(f"  - {ho_pkl.name}")


if __name__ == "__main__":
    main()
