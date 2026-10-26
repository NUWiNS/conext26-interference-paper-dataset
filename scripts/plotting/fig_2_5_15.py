#!/usr/bin/env python3
"""RSRP, Interference, and SINR CDF panels."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

OPS = ["att", "tmobile", "verizon"]
OP_LABELS = {
    "att": "AT&T",
    "tmobile": "T-Mobile",
    "verizon": "Verizon",
}
OP_COLORS = {
    "att": "royalblue",
    "tmobile": "magenta",
    "verizon": "red",
}

# Global font settings for all CDF figures
LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 11
TICK_FONTSIZE = 12
FONTWEIGHT = "bold"
FIG_SIZE = (4, 2.5)

RAT_ORDER = ["LTE", "NR"]
BAND_TYPE_ORDER = ["low", "mid", "mmwave", "unknown"]
PLOT_BAND_TYPE_ORDER = ["low", "mid", "mmwave"]
DEFAULT_UE_REPORT = (
    REPO_ROOT / "data/intermediate_data/2024/complete_ue_level_metrics_report.csv"
)
CACHE_VERSION = "overall-interference-complete-ue-v1"
CACHE_COLUMNS = {
    "operator",
    "rat",
    "interference_dbm",
    "rsrp_dbm",
    "sinr_db",
    "band",
    "channel_key",
    "frequency_mhz",
    "band_type",
    "cache_version",
}

NR_INT_COL = "NR KPI PCell Interference [dBm]"
LTE_INT_COL = "LTE KPI PCell Interference [dBm]"
NR_RSRP_COL = "5G KPI PCell RF Serving SS-RSRP [dBm]"
LTE_RSRP_COL = "LTE KPI PCell Serving RSRP [dBm]"
NR_SINR_COL = "5G KPI PCell RF Serving SS-SINR [dB]"
LTE_SINR_COL = "LTE KPI PCell SINR [dB]"
NR_BAND_COL = "5G KPI PCell RF Band"
LTE_BAND_COL = "LTE KPI PCell Serving Band"
NR_FREQ_COL = "5G KPI PCell RF Frequency [MHz]"
NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"
LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"
RAT_TO_INTERFERENCE_COL = {
    "LTE": LTE_INT_COL,
    "NR": NR_INT_COL,
}
RAT_TO_RSRP_COL = {
    "LTE": LTE_RSRP_COL,
    "NR": NR_RSRP_COL,
}
RAT_TO_SINR_COL = {
    "LTE": LTE_SINR_COL,
    "NR": NR_SINR_COL,
}
RAT_TO_BAND_COL = {
    "LTE": LTE_BAND_COL,
    "NR": NR_BAND_COL,
}
RAT_TO_CHANNEL_COL = {
    "LTE": LTE_EARFCN_COL,
    "NR": NR_FREQ_COL,
}

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

TECH_COLORS = {
    "LTE-low": "#08710C",
    "LTE-mid": "#70CA32",
    "5G-low": "#F3FF33",
    "5G-mid": "#FFB233",
    "5G-mmWave": "#CB0404",
    "unknown": "gray",
}
BAND_TYPE_LABELS = {
    "low": "low",
    "mid": "mid",
    "mmwave": "mmWave",
    "unknown": "unknown",
}
METRICS = {
    "interference": {
        "column": "interference_dbm",
        "label": "Interference",
        "xlabel": "PCell Interference (dBm)",
        "unit": "dBm",
        "xlim": (-120, -60),
    },
    "rsrp": {
        "column": "rsrp_dbm",
        "label": "RSRP",
        "xlabel": "PCell RSRP (dBm)",
        "unit": "dBm",
        "xlim": (-120, -60),
    },
    "sinr": {
        "column": "sinr_db",
        "label": "SINR",
        "xlabel": "PCell SINR (dB)",
        "unit": "dB",
        "xlim": (-10, 30),
    },
}


def install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias, target in aliases.items():
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(target)
        except ImportError:
            pass


def resolve_repo_path(path: Path) -> Path:
    """Resolve relative paths from the repo root, regardless of launch cwd."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def guess_op_from_name(s: str) -> Optional[str]:
    sl = str(s).lower()
    for op in OPS:
        if op in sl:
            return op
    if "tmobi" in sl:
        return "tmobile"
    return None


def clean_value(x) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]
    return s


def normalize_rat_label(x) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"LTE", "4G"}:
        return "LTE"
    if s in {"NR", "5G"}:
        return "NR"
    return None


def report_rat_to_cache_rat(x) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"LTE", "4G"}:
        return "LTE"
    if s in {"NR", "5G"}:
        return "NR"
    return None


def normalize_band_type(x) -> str:
    s = clean_value(x)
    if s is None:
        return "unknown"
    s = s.lower()
    if s in {"low", "mid", "mmwave"}:
        return s
    if s in {"mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return "unknown"


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
    earfcn_num = parse_float(earfcn)
    if earfcn_num is None:
        return None
    earfcn_int = int(round(earfcn_num))

    band_num = extract_band_number(band)
    candidate_tables = []
    if band_num in LTE_DL_BAND_TABLE:
        candidate_tables.append(LTE_DL_BAND_TABLE[band_num])
    else:
        candidate_tables.extend(LTE_DL_BAND_TABLE.values())

    for f_dl_low, n_offs_dl, n_min, n_max in candidate_tables:
        if n_min <= earfcn_int <= n_max:
            return f_dl_low + 0.1 * (earfcn_int - n_offs_dl)
    return None


def nr_arfcn_to_freq_mhz(arfcn) -> Optional[float]:
    arfcn_num = parse_float(arfcn)
    if arfcn_num is None:
        return None
    n_ref = int(round(arfcn_num))
    if 0 <= n_ref <= 599999:
        return 0.005 * n_ref
    if 600000 <= n_ref <= 2016666:
        return 3000.0 + 0.015 * (n_ref - 600000)
    if 2016667 <= n_ref <= 3279165:
        return 24250.08 + 0.06 * (n_ref - 2016667)
    return None


def band_type_from_freq_mhz(freq_mhz: Optional[float], rat: str) -> str:
    if freq_mhz is None or not np.isfinite(freq_mhz):
        return "unknown"
    if freq_mhz < 1000:
        return "low"
    if rat == "NR" and freq_mhz >= 24000:
        return "mmwave"
    return "mid"


def infer_lte_frequency_series(bands: pd.Series, earfcns: pd.Series) -> pd.Series:
    earfcn_num = pd.to_numeric(earfcns, errors="coerce").round()
    band_num = bands.astype("string").str.extract(r"(?:Band|b|n)\s*(\d+)", expand=False)
    band_num = pd.to_numeric(band_num, errors="coerce")
    freqs = pd.Series(np.nan, index=earfcns.index, dtype=float)

    for band, (f_dl_low, n_offs_dl, n_min, n_max) in LTE_DL_BAND_TABLE.items():
        in_range = earfcn_num.between(n_min, n_max)
        same_band = band_num.eq(band)
        unknown_band = band_num.isna()
        mask = freqs.isna() & in_range & (same_band | unknown_band)
        freqs.loc[mask] = f_dl_low + 0.1 * (earfcn_num.loc[mask] - n_offs_dl)

    return freqs


def infer_nr_frequency_series(freqs: pd.Series, arfcns: pd.Series) -> pd.Series:
    out = pd.to_numeric(freqs, errors="coerce")
    arfcn = pd.to_numeric(arfcns, errors="coerce").round()

    mask = out.isna() & arfcn.between(0, 599999)
    out.loc[mask] = 0.005 * arfcn.loc[mask]

    mask = out.isna() & arfcn.between(600000, 2016666)
    out.loc[mask] = 3000.0 + 0.015 * (arfcn.loc[mask] - 600000)

    mask = out.isna() & arfcn.between(2016667, 3279165)
    out.loc[mask] = 24250.08 + 0.06 * (arfcn.loc[mask] - 2016667)

    return out


def parse_operator_list(x) -> List[str]:
    ops: List[str] = []
    for part in str(x).split(","):
        op = guess_op_from_name(part.strip()) or part.strip().lower()
        if op in OPS and op not in ops:
            ops.append(op)
    return ops


def infer_supported_band_type_from_freq(
    freq_mhz: Optional[float], rat: str, supported_band_types: set
) -> str:
    inferred = band_type_from_freq_mhz(freq_mhz, rat)
    if inferred == "unknown":
        return "unknown"
    if supported_band_types and inferred not in supported_band_types:
        return "unknown"
    return inferred


def map_band_info_series(
    op: str,
    rat: str,
    bands: pd.Series,
    channels: pd.Series,
    freqs: pd.Series,
    band_type_map: Dict[Tuple[str, str, str], str],
    channel_info_map: Dict[Tuple[str, str, str], Tuple[str, str]],
    supported_band_types_map: Dict[Tuple[str, str], set],
) -> Tuple[pd.Series, pd.Series]:
    supported_band_types = supported_band_types_map.get((op, rat), set())
    out_bands = pd.Series(
        [
            band if band_type_map.get((op, rat, band)) is not None else "unknown"
            for band in bands
        ],
        index=bands.index,
        dtype=object,
    )
    band_lookup = pd.Series(
        [band_type_map.get((op, rat, band), "unknown") for band in bands],
        index=bands.index,
        dtype=object,
    )

    unknown = band_lookup.eq("unknown")
    if unknown.any():
        channel_bands = []
        channel_types = []
        for channel in channels.loc[unknown]:
            band, band_type = channel_info_map.get(
                (op, rat, channel), ("unknown", "unknown")
            )
            channel_bands.append(band)
            channel_types.append(band_type)
        channel_band_lookup = pd.Series(
            channel_bands, index=channels.loc[unknown].index, dtype=object
        )
        channel_lookup = pd.Series(
            channel_types, index=channels.loc[unknown].index, dtype=object
        )
        matched = channel_lookup.ne("unknown")
        out_bands.loc[channel_lookup.index[matched]] = channel_band_lookup.loc[matched]
        band_lookup.loc[unknown] = channel_lookup

    unknown = band_lookup.eq("unknown")
    if unknown.any():
        freq_vals = pd.to_numeric(freqs, errors="coerce")
        inferred = pd.Series(
            [
                infer_supported_band_type_from_freq(freq, rat, supported_band_types)
                for freq in freq_vals.loc[unknown]
            ],
            index=freq_vals.loc[unknown].index,
            dtype=object,
        )
        band_lookup.loc[unknown] = inferred

    return out_bands.fillna("unknown"), band_lookup.fillna("unknown")


def load_band_type_maps(
    path: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], str],
    Dict[Tuple[str, str, str], Tuple[str, str]],
    Dict[Tuple[str, str], set],
]:
    if not path.exists():
        print(f"[WARN] Band summary not found: {path}")
        return {}, {}, {}

    df = pd.read_csv(path)
    required = {"rat", "band", "band_type", "channel_keys", "operators"}
    missing = required.difference(df.columns)
    if missing:
        print(f"[WARN] Band summary missing columns {sorted(missing)}: {path}")
        return {}, {}, {}

    band_map: Dict[Tuple[str, str, str], str] = {}
    channel_map: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    supported_band_types_map: Dict[Tuple[str, str], set] = {}
    for _, row in df.iterrows():
        rat = normalize_rat_label(row.get("rat"))
        band = clean_value(row.get("band"))
        band_type = normalize_band_type(row.get("band_type"))
        operators = parse_operator_list(row.get("operators"))
        if rat is None or not operators:
            continue
        for op in operators:
            supported_band_types_map.setdefault((op, rat), set()).add(band_type)
            if band is not None:
                band_map[(op, rat, band)] = band_type
            for channel in str(row.get("channel_keys", "")).split(","):
                channel = clean_value(channel)
                if channel is not None:
                    channel_map[(op, rat, channel)] = (band or "unknown", band_type)
    return band_map, channel_map, supported_band_types_map


def pick_operator_pcell_pkl(pkl_root: Path, op: str) -> Optional[Path]:
    exact = pkl_root / f"{op}_pcell.pkl"
    if exact.exists():
        return exact

    files = sorted(pkl_root.rglob("*_pcell.pkl"))
    cands = [p for p in files if guess_op_from_name(p.name) == op]
    return cands[0] if cands else None


def build_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    v = np.sort(v)
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def write_cdf_stats(
    out_path: Path,
    title: str,
    unit: str,
    series: List[Tuple[str, np.ndarray]],
) -> None:
    percentiles = [5, 25, 50, 75, 95]
    lines = [title, "=" * len(title), ""]
    for label, values in series:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        lines.append(f"{label} (count={v.size})")
        if v.size == 0:
            lines.append("  no valid samples")
            lines.append("")
            continue
        lines.append(f"  min: {float(np.min(v)):.4f} {unit}")
        for p in percentiles:
            lines.append(f"  {p}%: {float(np.percentile(v, p)):.4f} {unit}")
        lines.append(f"  max: {float(np.max(v)):.4f} {unit}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[OK] Wrote {out_path}")


def _cache_file_pattern(op: str, rat: str) -> str:
    return f"overall_interference_{op}_{rat.lower()}*.csv"


def cache_files_for(out_dir: Path, op: str, rat: str) -> List[Path]:
    pat = _cache_file_pattern(op, rat)
    files = sorted(out_dir.glob(pat))
    exact_name = f"overall_interference_{op}_{rat.lower()}.csv"
    return [
        f
        for f in files
        if f.name == exact_name
        or re.fullmatch(
            rf"overall_interference_{re.escape(op)}_{rat.lower()}_part\d+\.csv",
            f.name,
        )
    ]


def remove_cache_files(out_dir: Path, op: str, rat: str) -> None:
    for path in cache_files_for(out_dir, op, rat):
        path.unlink()


def split_and_write_csv_by_size(
    df: pd.DataFrame, out_dir: Path, stem: str, max_bytes: int
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out_path = out_dir / f"{stem}.csv"
        df.to_csv(out_path, index=False)
        return [out_path]

    sample_n = min(len(df), 1000)
    sample_csv = df.head(sample_n).to_csv(index=False)
    avg_row_bytes = max(1, len(sample_csv.encode("utf-8")) / max(1, sample_n))
    rows_per_part = max(1, int(max_bytes / avg_row_bytes))

    if len(df) <= rows_per_part:
        out_path = out_dir / f"{stem}.csv"
        df.to_csv(out_path, index=False)
        return [out_path]

    out_paths: List[Path] = []
    for i, start in enumerate(range(0, len(df), rows_per_part), start=1):
        out_path = out_dir / f"{stem}_part{i:02d}.csv"
        df.iloc[start : start + rows_per_part].to_csv(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def extract_operator_interference(
    pkl_path: Path,
    op: str,
    band_type_map: Dict[Tuple[str, str, str], str],
    channel_info_map: Dict[Tuple[str, str, str], Tuple[str, str]],
    supported_band_types_map: Dict[Tuple[str, str], set],
) -> Dict[str, pd.DataFrame]:
    install_numpy_pickle_compat()
    try:
        payload = pd.read_pickle(pkl_path)
    except Exception as exc:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        raise RuntimeError(
            f"Could not read {pkl_path.name} with pandas {pd.__version__}. "
            f"If this is the system Python, retry with {venv_python}."
        ) from exc

    if not isinstance(payload, dict) or "df" not in payload:
        raise TypeError(f"Expected dict payload with df in {pkl_path}")

    df = payload["df"]
    out: Dict[str, pd.DataFrame] = {}

    for rat, col in RAT_TO_INTERFERENCE_COL.items():
        band_col = RAT_TO_BAND_COL[rat]
        channel_col = RAT_TO_CHANNEL_COL[rat]
        rsrp_col = RAT_TO_RSRP_COL[rat]
        sinr_col = RAT_TO_SINR_COL[rat]
        if col not in df.columns:
            print(f"[WARN] {op}: missing column for {rat}: {col}")
            out[rat] = pd.DataFrame(columns=sorted(CACHE_COLUMNS))
            continue
        if band_col not in df.columns:
            print(f"[WARN] {op}: missing band column for {rat}: {band_col}")
        if channel_col not in df.columns:
            print(f"[WARN] {op}: missing channel column for {rat}: {channel_col}")

        values = pd.to_numeric(df[col], errors="coerce")
        valid = np.isfinite(values)
        values = values[valid]
        rsrp = (
            pd.to_numeric(df.loc[valid, rsrp_col], errors="coerce")
            if rsrp_col in df.columns
            else pd.Series(np.nan, index=values.index)
        )
        sinr = (
            pd.to_numeric(df.loc[valid, sinr_col], errors="coerce")
            if sinr_col in df.columns
            else pd.Series(np.nan, index=values.index)
        )
        if band_col in df.columns:
            bands = df.loc[valid, band_col].map(clean_value)
        else:
            bands = pd.Series(["unknown"] * len(values), index=values.index)
        if channel_col in df.columns:
            channels = df.loc[valid, channel_col].map(clean_value)
        else:
            channels = pd.Series(["unknown"] * len(values), index=values.index)

        if rat == "LTE":
            frequencies = infer_lte_frequency_series(bands, channels)
        else:
            nr_arfcn = (
                df.loc[valid, NR_ARFCN_COL].map(clean_value)
                if NR_ARFCN_COL in df.columns
                else pd.Series([pd.NA] * len(values), index=values.index)
            )
            frequencies = infer_nr_frequency_series(channels, nr_arfcn)

        matched_bands, band_types = map_band_info_series(
            op,
            rat,
            bands,
            channels,
            frequencies,
            band_type_map,
            channel_info_map,
            supported_band_types_map,
        )

        op_df = pd.DataFrame(
            {
                "operator": op,
                "rat": rat,
                "interference_dbm": values.astype(float).to_numpy(),
                "rsrp_dbm": rsrp.to_numpy(),
                "sinr_db": sinr.to_numpy(),
                "band": matched_bands.to_numpy(),
                "channel_key": channels.fillna("unknown").to_numpy(),
                "frequency_mhz": frequencies.to_numpy(),
                "band_type": band_types.to_numpy(),
                "cache_version": CACHE_VERSION,
            }
        )
        out[rat] = op_df

    return out


def extract_operator_interference_from_ue_report(
    ue_report: Path,
    op: str,
    band_type_map: Dict[Tuple[str, str, str], str],
    channel_info_map: Dict[Tuple[str, str, str], Tuple[str, str]],
    supported_band_types_map: Dict[Tuple[str, str], set],
    chunksize: int = 500_000,
) -> Dict[str, pd.DataFrame]:
    if not ue_report.exists():
        raise FileNotFoundError(f"Complete UE-level report not found: {ue_report}")

    required_cols = {
        "operator",
        "rat",
        "interference",
        "rsrp",
        "sinr",
        "band",
        "channel_key",
        "frequency_mhz",
        "band_type",
    }
    parts: Dict[str, List[pd.DataFrame]] = {rat: [] for rat in RAT_ORDER}
    saw_columns = None

    for chunk in pd.read_csv(
        ue_report, usecols=lambda c: c in required_cols, chunksize=chunksize
    ):
        if saw_columns is None:
            saw_columns = set(chunk.columns)
            missing = required_cols.difference(saw_columns)
            if missing:
                raise ValueError(
                    f"{ue_report} is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )

        mask_op = chunk["operator"].astype(str).str.lower().eq(op)
        if not mask_op.any():
            continue

        work = chunk.loc[mask_op].copy()
        work["rat_cache"] = work["rat"].map(report_rat_to_cache_rat)
        work = work[work["rat_cache"].isin(RAT_ORDER)]
        if work.empty:
            continue

        for rat in RAT_ORDER:
            sub = work[work["rat_cache"].eq(rat)].copy()
            if sub.empty:
                continue

            values = pd.to_numeric(sub["interference"], errors="coerce")
            valid = np.isfinite(values)
            if not valid.any():
                continue
            sub = sub.loc[valid].copy()
            values = values.loc[valid]

            rsrp = pd.to_numeric(sub["rsrp"], errors="coerce")
            sinr = pd.to_numeric(sub["sinr"], errors="coerce")
            bands = sub["band"].map(clean_value).fillna("unknown")
            channels = sub["channel_key"].map(clean_value).fillna("unknown")
            frequencies = pd.to_numeric(sub["frequency_mhz"], errors="coerce")

            matched_bands, mapped_band_types = map_band_info_series(
                op,
                rat,
                bands,
                channels,
                frequencies,
                band_type_map,
                channel_info_map,
                supported_band_types_map,
            )
            source_band_types = sub["band_type"].map(normalize_band_type)
            band_types = mapped_band_types.where(
                mapped_band_types.ne("unknown"), source_band_types
            )

            parts[rat].append(
                pd.DataFrame(
                    {
                        "operator": op,
                        "rat": rat,
                        "interference_dbm": values.astype(float).to_numpy(),
                        "rsrp_dbm": rsrp.to_numpy(),
                        "sinr_db": sinr.to_numpy(),
                        "band": matched_bands.to_numpy(),
                        "channel_key": channels.to_numpy(),
                        "frequency_mhz": frequencies.to_numpy(),
                        "band_type": band_types.to_numpy(),
                        "cache_version": CACHE_VERSION,
                    }
                )
            )

    return {
        rat: (
            pd.concat(rat_parts, ignore_index=True)
            if rat_parts
            else pd.DataFrame(columns=sorted(CACHE_COLUMNS))
        )
        for rat, rat_parts in parts.items()
    }


def load_cache_parts(paths: Iterable[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:
            print(f"[WARN] Could not load cache {path.name}: {exc}")
    if not dfs:
        return pd.DataFrame(columns=sorted(CACHE_COLUMNS))
    return pd.concat(dfs, ignore_index=True)


def load_or_build_operator_cache(
    op: str,
    ue_report: Path,
    out_dir: Path,
    max_cache_bytes: int,
    force_rebuild: bool,
    band_type_map: Dict[Tuple[str, str, str], str],
    channel_info_map: Dict[Tuple[str, str, str], Tuple[str, str]],
    supported_band_types_map: Dict[Tuple[str, str], set],
) -> Dict[str, pd.DataFrame]:
    cached: Dict[str, pd.DataFrame] = {}
    missing_rats = []

    for rat in RAT_ORDER:
        paths = cache_files_for(out_dir, op, rat)
        if paths and not force_rebuild:
            cache_df = load_cache_parts(paths)
            missing_cols = CACHE_COLUMNS.difference(cache_df.columns)
            if missing_cols:
                print(
                    f"[INFO] {op} {rat}: rebuilding old cache schema; "
                    f"missing {sorted(missing_cols)}"
                )
                missing_rats.append(rat)
            elif set(
                cache_df.get("cache_version", pd.Series(dtype=object)).dropna().unique()
            ) != {CACHE_VERSION}:
                print(f"[INFO] {op} {rat}: rebuilding old band-match cache")
                missing_rats.append(rat)
            else:
                cached[rat] = cache_df
                print(f"[OK] {op} {rat}: loaded {len(paths)} cache file(s)")
        else:
            missing_rats.append(rat)

    if not missing_rats:
        return cached

    print(f"[INFO] {op}: reading {ue_report.name} to build cache")
    extracted = extract_operator_interference_from_ue_report(
        ue_report, op, band_type_map, channel_info_map, supported_band_types_map
    )

    for rat in RAT_ORDER:
        if rat not in missing_rats:
            continue
        remove_cache_files(out_dir, op, rat)
        stem = f"overall_interference_{op}_{rat.lower()}"
        part_files = split_and_write_csv_by_size(
            extracted[rat], out_dir, stem=stem, max_bytes=max_cache_bytes
        )
        cached[rat] = extracted[rat]
        print(
            f"[OK] {op} {rat}: rows={len(extracted[rat]):,}, "
            f"cache_parts={len(part_files)}"
        )

    return cached


def metric_output_stem(metric_name: str) -> str:
    if metric_name == "interference":
        return "overall_interference"
    return f"overall_{metric_name}"


def plot_rat_metric_cdf(
    metric_name: str,
    rat: str,
    rat_df: pd.DataFrame,
    operators: List[str],
    out_path: Path,
) -> None:
    metric = METRICS[metric_name]
    metric_col = metric["column"]
    sub = rat_df.loc[rat_df["rat"].astype(str).str.upper() == rat].copy()
    if sub.empty:
        print(f"[WARN] No data to plot for {rat} {metric_name}")
        return

    plt.figure(figsize=FIG_SIZE)

    stats_series: List[Tuple[str, np.ndarray]] = []
    for op in operators:
        vals = pd.to_numeric(
            sub.loc[
                sub["operator"].astype(str).str.lower() == op,
                metric_col,
            ],
            errors="coerce",
        ).dropna()
        op_label = OP_LABELS.get(op, op.upper())
        vals_np = vals.to_numpy(dtype=float)
        stats_series.append((op_label, vals_np))
        xs, ys = build_cdf(vals_np)
        if xs.size == 0:
            print(f"[WARN] No valid {rat} {metric_name} values for {op}")
            continue

        plt.plot(
            xs,
            ys,
            color=OP_COLORS.get(op),
            linewidth=2.5,
            label=f"{op_label}",
        )

    pretty_rat = "5G" if rat == "NR" else "LTE"
    plt.xlabel(metric["xlabel"], fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.ylabel("CDF", fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.xticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    plt.yticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    if metric["xlim"] is not None:
        plt.xlim(*metric["xlim"])
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if metric_name == "rsrp":
        plt.legend(prop={"size": LEGEND_FONTSIZE, "weight": FONTWEIGHT})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_path}")

    if out_path.suffix.lower() == ".png":
        write_cdf_stats(
            out_path.with_suffix(".txt"),
            title=f"{pretty_rat} Overall PCell {metric['label']} CDF Statistics",
            unit=metric["xlabel"],
            series=stats_series,
        )


def plot_operator_rat_band_type_metric_cdf(
    metric_name: str,
    op: str,
    rat: str,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    metric = METRICS[metric_name]
    metric_col = metric["column"]
    sub = df.loc[
        (df["operator"].astype(str).str.lower() == op)
        & (df["rat"].astype(str).str.upper() == rat)
    ].copy()
    if sub.empty:
        print(f"[WARN] No data to plot for {op} {rat} {metric_name} by band type")
        return

    plt.figure(figsize=FIG_SIZE)
    plotted = False
    for band_type in PLOT_BAND_TYPE_ORDER:
        vals = pd.to_numeric(
            sub.loc[
                sub["band_type"].astype(str).str.lower() == band_type,
                metric_col,
            ],
            errors="coerce",
        ).dropna()
        xs, ys = build_cdf(vals.to_numpy(dtype=float))
        if xs.size == 0:
            continue

        label_rat = "5G" if rat == "NR" else "LTE"
        label = f"{label_rat}-{BAND_TYPE_LABELS.get(band_type, band_type)}"
        plt.plot(
            xs,
            ys,
            color=TECH_COLORS.get(label, TECH_COLORS["unknown"]),
            linestyle="-",
            linewidth=2.5,
            label=label,
        )
        plotted = True

    if not plotted:
        print(f"[WARN] No valid {op} {rat} {metric_name} band-type CDF data")
        plt.close()
        return

    plt.xlabel(metric["xlabel"], fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.ylabel("CDF", fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.xticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    plt.yticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    if metric["xlim"] is not None:
        plt.xlim(*metric["xlim"])
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if metric_name == "rsrp":
        plt.legend(prop={"size": LEGEND_FONTSIZE, "weight": FONTWEIGHT})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_path}")


def plot_band_type_operator_metric_cdf(
    metric_name: str,
    rat: str,
    band_type: str,
    df: pd.DataFrame,
    operators: List[str],
    out_path: Path,
) -> None:
    metric = METRICS[metric_name]
    metric_col = metric["column"]
    sub = df.loc[
        (df["rat"].astype(str).str.upper() == rat)
        & (df["band_type"].astype(str).str.lower() == band_type)
    ].copy()
    if sub.empty:
        return

    plt.figure(figsize=FIG_SIZE)
    plotted = False
    stats_series: List[Tuple[str, np.ndarray]] = []
    for op in operators:
        vals = pd.to_numeric(
            sub.loc[
                sub["operator"].astype(str).str.lower() == op,
                metric_col,
            ],
            errors="coerce",
        ).dropna()
        op_label = OP_LABELS.get(op, op.upper())
        vals_np = vals.to_numpy(dtype=float)
        stats_series.append((op_label, vals_np))
        xs, ys = build_cdf(vals_np)
        if xs.size == 0:
            continue
        plt.plot(
            xs,
            ys,
            color=OP_COLORS.get(op),
            linewidth=2.5,
            label=op_label,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    pretty_rat = "5G" if rat == "NR" else "LTE"
    pretty_band_type = BAND_TYPE_LABELS.get(band_type, band_type)
    plt.xlabel(metric["xlabel"], fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.ylabel("CDF", fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.xticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    plt.yticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    if metric["xlim"] is not None:
        plt.xlim(*metric["xlim"])
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if metric_name == "rsrp":
        plt.legend(prop={"size": LEGEND_FONTSIZE, "weight": FONTWEIGHT})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_path}")

    if out_path.suffix.lower() == ".png":
        write_cdf_stats(
            out_path.with_suffix(".txt"),
            title=(
                f"{pretty_rat}-{pretty_band_type} PCell {metric['label']} "
                "by Operator CDF Statistics"
            ),
            unit=metric["xlabel"],
            series=stats_series,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot overall LTE and NR PCell metrics from complete UE-level report"
    )
    ap.add_argument(
        "--ue-report",
        type=Path,
        default=DEFAULT_UE_REPORT,
        help="Complete UE-level metrics CSV to use as the data source",
    )
    ap.add_argument(
        "--band-summary-csv",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/all_cells_band_summary.csv",
    )
    ap.add_argument(
        "--cache-dir", type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/overall_interference",
        help="Location for generated split CSVs and the summary CSV",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plots",
    )
    ap.add_argument("--operators", nargs="+", default=OPS)
    ap.add_argument(
        "--max-cache-bytes",
        type=int,
        default=50_000_000,
        help="Approximate maximum size per split cache CSV",
    )
    ap.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild split CSV caches even when they already exist",
    )
    args = ap.parse_args()

    ue_report = resolve_repo_path(args.ue_report)
    band_summary_csv = resolve_repo_path(args.band_summary_csv)
    cache_dir = resolve_repo_path(args.cache_dir)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    band_type_map, channel_info_map, supported_band_types_map = load_band_type_maps(
        band_summary_csv
    )

    operators = [guess_op_from_name(op) or str(op).lower() for op in args.operators]
    operators = [op for op in operators if op in OPS]
    if not operators:
        raise SystemExit("No supported operators requested")

    all_rows = []
    for op in operators:
        op_data = load_or_build_operator_cache(
            op=op,
            ue_report=ue_report,
            out_dir=cache_dir,
            max_cache_bytes=max(1, int(args.max_cache_bytes)),
            force_rebuild=bool(args.force_rebuild),
            band_type_map=band_type_map,
            channel_info_map=channel_info_map,
            supported_band_types_map=supported_band_types_map,
        )
        for rat in RAT_ORDER:
            if rat in op_data and not op_data[rat].empty:
                all_rows.append(op_data[rat])

    if not all_rows:
        raise SystemExit("No interference data found")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary = (
        all_df.groupby(["operator", "rat", "band", "band_type"])
        .agg(
            count_interference=("interference_dbm", "count"),
            median_interference=("interference_dbm", "median"),
            mean_interference=("interference_dbm", "mean"),
            count_rsrp=("rsrp_dbm", "count"),
            median_rsrp=("rsrp_dbm", "median"),
            mean_rsrp=("rsrp_dbm", "mean"),
            count_sinr=("sinr_db", "count"),
            median_sinr=("sinr_db", "median"),
            mean_sinr=("sinr_db", "mean"),
        )
        .reset_index()
    )
    summary_csv = cache_dir / "overall_interference_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote {summary_csv}")

    for letter, metric_name, rat in (
        ("a", "rsrp", "LTE"), ("b", "interference", "LTE"), ("c", "sinr", "LTE"),
        ("d", "rsrp", "NR"), ("e", "interference", "NR"), ("f", "sinr", "NR"),
    ):
        plot_rat_metric_cdf(metric_name, rat, all_df, operators, out_dir / f"fig_2{letter}.pdf")

    for figure, band_type in ((5, "low"), (5, "mid"), (15, "mmwave")):
        for index, metric_name in enumerate(("rsrp", "interference", "sinr")):
            letter = "abcdef"[(0 if band_type in ("low", "mmwave") else 3) + index]
            plot_band_type_operator_metric_cdf(
                metric_name, "NR", band_type, all_df, operators,
                out_dir / f"fig_{figure}{letter}.pdf",
            )


if __name__ == "__main__":
    main()
