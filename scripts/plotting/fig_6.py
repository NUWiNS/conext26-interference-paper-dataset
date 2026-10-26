#!/usr/bin/env python3
"""5G-low and 5G-mid boxplots"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_FONT_SIZE = 19
AXIS_FONT_SIZE = 17
LEGEND_FONT_SIZE = 16
FIG_SIZE = (5, 5)
FIG_SIZE_NR_MID = (5, 5)
FIG_SIZE_NR_LOW_MMWAVE = (4, 5)


def pick_fig_size(rat: str, band_type: str) -> Tuple[float, float]:
    if str(rat).upper() == "NR":
        if str(band_type).lower() == "mid":
            return FIG_SIZE_NR_MID
        if str(band_type).lower() in {"low", "mmwave"}:
            return FIG_SIZE_NR_LOW_MMWAVE
    return FIG_SIZE

plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": AXIS_FONT_SIZE,
        "ytick.labelsize": AXIS_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

PCELL_PKL_RE = re.compile(r"^(?P<op>[a-z]+)_pcell\.pkl$", re.IGNORECASE)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL_REPORT = ROOT / "data/intermediate_data/2024/all_cells_aggregated_report.csv"
DEFAULT_FIG45_REPORT = ROOT / "data/intermediate_data/2024/fig45_cell_metrics_report.csv"
DEFAULT_UE_RECORD_REPORT = ROOT / "data/intermediate_data/2024/complete_ue_level_metrics_report.csv"
OP_INITIALS = {"att": "A", "tmobile": "T", "verizon": "V"}
INITIAL_TO_OP = {v: k for k, v in OP_INITIALS.items()}
BAND_TYPE_ORDER = ["low", "mid", "mmwave"]
BAND_TYPE_LABELS = {"low": "low", "mid": "mid", "mmwave": "mmWave"}
MIN_CELLS_PER_UE_BAND = 15
METRICS = {
    "interference": {
        "column": "cell_mean_interference",
        "label": "Per-cell mean Interference [dBm]  (RSRP - SINR)",
        "title": "PCell mean Interference",
    },
    "rsrp": {
        "column": "cell_mean_rsrp",
        "label": "Per-cell mean RSRP [dBm]",
        "title": "PCell mean RSRP",
    },
    "sinr": {
        "column": "cell_mean_sinr",
        "label": "Per-cell mean SINR [dB]",
        "title": "PCell mean SINR",
    },
}

NR_PREFIX = "5G KPI "
LTE_PREFIX = "LTE KPI "


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def haversine_miles(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    R = 3958.7613  # miles
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def pick_lat_lon_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat_candidates = [
        "Lat",
        "Latitude",
        "lat",
        "latitude",
        "QUALCOMM Latitude",
        "GPS Latitude",
    ]
    lon_candidates = [
        "Lon",
        "Longitude",
        "lng",
        "lon",
        "longitude",
        "QUALCOMM Longitude",
        "GPS Longitude",
    ]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    return lat_col, lon_col


def build_non_city_mask(
    df: pd.DataFrame, cities_df: pd.DataFrame, radius_miles: float
) -> pd.Series:
    lat_col, lon_col = pick_lat_lon_cols(df)
    if lat_col is None or lon_col is None:
        return pd.Series(True, index=df.index)

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    keep = pd.Series(True, index=df.index)
    gps_ok = lat.notna() & lon.notna()
    if gps_ok.sum() == 0:
        return keep

    city_lat = pd.to_numeric(cities_df["lat"], errors="coerce").to_numpy(dtype=float)
    city_lon = pd.to_numeric(cities_df["lng"], errors="coerce").to_numpy(dtype=float)
    good = ~np.isnan(city_lat) & ~np.isnan(city_lon)
    city_lat = city_lat[good]
    city_lon = city_lon[good]
    if city_lat.size == 0:
        return keep

    pts_lat = lat[gps_ok].to_numpy(dtype=float)
    pts_lon = lon[gps_ok].to_numpy(dtype=float)
    d = haversine_miles(
        pts_lat[:, None], pts_lon[:, None], city_lat[None, :], city_lon[None, :]
    )
    keep_vals = d.min(axis=1) > radius_miles
    keep.loc[gps_ok] = keep_vals
    return keep


def load_pcell_pickle(path: Path) -> dict:
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def infer_rat_from_key(key: str) -> Optional[str]:
    if not isinstance(key, str):
        return None
    parts = key.split("|")
    if not parts:
        return None
    rat = parts[0].strip().upper()
    return rat if rat in ("NR", "LTE") else None


def build_rat_to_pcell_interference_col(components: List[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in components:
        try:
            rat = str(c["rat"]).upper()
            role = str(c["role"])
            idx = c.get("idx", None)
            inter_col = c.get("interference_col", None)
        except Exception:
            continue
        if role == "PCell" and idx is None and isinstance(inter_col, str):
            out[rat] = inter_col
    return out


def normalize_band(x: Any, rat: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    s_low = s.lower().replace(" ", "")
    # Common patterns: "n77", "NR n77", "Band n77", "77" etc.
    if rat == "NR":
        m = re.search(r"n(\d+)", s_low)
        if m:
            return f"n{m.group(1)}"
        m2 = re.search(r"\b(\d{1,3})\b", s_low)
        if m2:
            return f"n{m2.group(1)}"
        return s  # fallback
    else:
        m = re.search(r"b(\d+)", s_low)
        if m:
            return f"b{m.group(1)}"
        m2 = re.search(r"\b(\d{1,3})\b", s_low)
        if m2:
            return f"b{m2.group(1)}"
        return s  # fallback


def normalize_report_rat(x: Any) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"NR", "5G"}:
        return "NR"
    if s in {"LTE", "4G"}:
        return "LTE"
    return None


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def collect_band_distributions_from_cell_report(
    report_csv: Path,
    operators: List[str],
) -> pd.DataFrame:
    required_cols = {
        "operator",
        "rat",
        "band",
        "pcell_key",
        "avg_rsrp",
        "avg_interference",
        "avg_sinr",
    }
    df = pd.read_csv(report_csv)
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"{report_csv} is missing required columns: {', '.join(sorted(missing))}"
        )

    n_samples_col = first_existing_column(df, ["n_matched_rows", "n_rows"])
    pkl_col = first_existing_column(df, ["pkl", "pcell_pickle"])
    op_set = {o.lower() for o in operators}
    out = pd.DataFrame(
        {
            "operator": df["operator"].astype(str).str.lower(),
            "rat": df["rat"].map(normalize_report_rat),
            "pcell_key": df["pcell_key"].astype(str),
            "band": df.apply(
                lambda r: normalize_band(
                    r["band"], normalize_report_rat(r["rat"]) or ""
                ),
                axis=1,
            ),
            "cell_mean_interference": pd.to_numeric(
                df["avg_interference"], errors="coerce"
            ),
            "cell_mean_rsrp": pd.to_numeric(df["avg_rsrp"], errors="coerce"),
            "cell_mean_sinr": pd.to_numeric(df["avg_sinr"], errors="coerce"),
            "n_valid_samples": (
                pd.to_numeric(df[n_samples_col], errors="coerce")
                if n_samples_col is not None
                else pd.Series(np.nan, index=df.index)
            ),
            "pcell_pickle": (df[pkl_col].astype(str) if pkl_col is not None else ""),
        }
    )
    out["day"] = 0
    out["band_col"] = f"{report_csv.name}.band"
    out["interference_col"] = f"{report_csv.name}.avg_interference"
    optional_cols = [
        "avg_interferers",
        "frequency_label",
        "band_type_group_label",
        "band_type",
        "area_type",
        "period",
    ]
    for col in optional_cols:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = np.nan
    out = out[
        out["operator"].isin(op_set)
        & out["rat"].isin(["NR", "LTE"])
        & out["band"].notna()
        & (
            out["cell_mean_interference"].notna()
            | out["cell_mean_rsrp"].notna()
            | out["cell_mean_sinr"].notna()
        )
    ].copy()
    out["n_valid_samples"] = out["n_valid_samples"].fillna(0).astype(int)
    return out[
        [
            "operator",
            "day",
            "rat",
            "pcell_key",
            "band",
            "cell_mean_interference",
            "cell_mean_rsrp",
            "cell_mean_sinr",
            "n_valid_samples",
            "band_col",
            "interference_col",
            "pcell_pickle",
            "avg_interferers",
            "frequency_label",
            "band_type_group_label",
            "band_type",
            "area_type",
            "period",
        ]
    ]


def pick_pcell_band_col(df: pd.DataFrame, rat: str) -> Optional[str]:
    prefix = NR_PREFIX if rat == "NR" else LTE_PREFIX
    candidates = [
        c
        for c in df.columns
        if str(c).startswith(prefix)
        and "PCell" in str(c)
        and "Width" not in str(c)
        and ("Band" in str(c) or "band" in str(c))
    ]

    if not candidates:
        return None

    def score(col: str) -> Tuple[int, int, int]:
        s = str(col).lower()
        return (
            1 if "band" in s else 0,
            1 if "rf" in s else 0,
            1 if "serv" in s else 0,
        )

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def find_pcell_pickles(pickle_dir: Path) -> List[Path]:
    return sorted(pickle_dir.glob("*_pcell.pkl"))


def collect_band_distributions(
    pickle_paths: List[Path],
    operators: List[str],
    cities_df: Optional[pd.DataFrame],
    city_radius_miles: float,
    min_valid_samples_per_cell: int = 3,
) -> pd.DataFrame:
    """
    Build per-cell average interference records:
      columns: operator, day, rat, pcell_key, band, cell_mean_interference, n_valid, pcell_pickle
    """
    rows: List[Dict[str, Any]] = []
    op_set = {o.lower() for o in operators}

    for pkl in pickle_paths:
        m = PCELL_PKL_RE.match(pkl.name)
        op_from_name = m.group("op").lower()
        payload = load_pcell_pickle(pkl)
        meta = payload.get("meta", {})
        op = str(meta.get("operator", op_from_name)).lower()
        if op not in op_set:
            continue

        day = meta.get("day", 0)

        df: pd.DataFrame = payload["df"]
        pcell_index: Dict[str, np.ndarray] = payload["pcell_index"]
        components = payload.get("components", [])
        rat2inter = build_rat_to_pcell_interference_col(components)

        # City exclusion mask for this df (optional)
        if cities_df is not None:
            non_city = build_non_city_mask(df, cities_df, city_radius_miles)
        else:
            non_city = pd.Series(True, index=df.index)

        # Band column discovery per RAT
        band_col_nr = pick_pcell_band_col(df, "NR")
        band_col_lte = pick_pcell_band_col(df, "LTE")

        for key, idxs in pcell_index.items():
            rat = infer_rat_from_key(key)
            if rat not in ("NR", "LTE"):
                continue

            inter_col = rat2inter.get(rat)
            if not inter_col or inter_col not in df.columns:
                continue

            band_col = band_col_nr if rat == "NR" else band_col_lte
            if band_col is None or band_col not in df.columns:
                continue

            sub = df.loc[idxs]
            # apply city exclusion
            sub = sub.loc[non_city.loc[sub.index].to_numpy()]
            if sub.empty:
                continue

            inter = pd.to_numeric(sub[inter_col], errors="coerce")
            good = inter.notna()
            if good.sum() < min_valid_samples_per_cell:
                continue

            sub_good = sub.loc[good]
            cell_mean = float(
                pd.to_numeric(sub_good[inter_col], errors="coerce").mean()
            )

            bseries = sub_good[band_col].astype("string")
            bseries = bseries.replace({"": pd.NA})
            if bseries.notna().sum() == 0:
                continue
            band_mode = bseries.mode(dropna=True)
            band_val = band_mode.iloc[0] if len(band_mode) else bseries.dropna().iloc[0]
            band = normalize_band(band_val, rat)
            if band is None:
                continue

            rows.append(
                {
                    "operator": op,
                    "day": int(day) if str(day).isdigit() else day,
                    "rat": rat,
                    "pcell_key": key,
                    "band": band,
                    "cell_mean_interference": cell_mean,
                    "cell_mean_rsrp": np.nan,
                    "cell_mean_sinr": np.nan,
                    "n_valid_samples": int(good.sum()),
                    "band_col": band_col,
                    "interference_col": inter_col,
                    "pcell_pickle": pkl.name,
                    "avg_interferers": np.nan,
                    "frequency_label": np.nan,
                    "band_type_group_label": np.nan,
                    "band_type": np.nan,
                    "area_type": np.nan,
                    "period": np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_operator_rat(
    df_cells: pd.DataFrame,
    operator: str,
    rat: str,
    out_dir: Path,
    metric_name: str,
    metric_col: str,
    metric_label: str,
    metric_title: str,
) -> None:
    """
    Make one plot: CDF curves per band for a given operator+rat+metric.
    """
    sub = df_cells[
        (df_cells["operator"].str.lower() == operator.lower())
        & (df_cells["rat"] == rat)
    ]
    if sub.empty:
        print(f"[WARN] No data for {operator} {rat} {metric_name}")
        return
    if metric_col not in sub.columns:
        print(f"[WARN] Missing {metric_col} for {operator} {rat}")
        return
    sub = sub[sub[metric_col].notna()].copy()
    if sub.empty:
        print(f"[WARN] No non-NaN {metric_name} data for {operator} {rat}")
        return

    min_points_per_band = 10
    band_counts = sub["band"].value_counts()
    bands = band_counts[band_counts >= min_points_per_band].index.tolist()
    if not bands:
        print(
            f"[WARN] No bands with >= {min_points_per_band} points for {operator} {rat}"
        )
        return

    bands.sort(key=lambda b: (sub[sub["band"] == b].shape[0]), reverse=True)

    plt.figure()
    for b in bands:
        vals = sub.loc[sub["band"] == b, metric_col].to_numpy(dtype=float)
        xs, ys = ecdf(vals)
        if xs.size == 0:
            continue
        plt.plot(xs, ys, label=f"{b} (n={len(vals)})")

    plt.xlabel(metric_label)
    plt.ylabel("CDF")
    plt.title(f"{operator.upper()} — {rat} {metric_title} by Band")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / f"{operator}_{rat}_band_{metric_name}_cdf.png"
    plt.savefig(out_path, dpi=200)
    out_path = out_dir / f"{operator}_{rat}_band_{metric_name}_cdf.pdf"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Wrote {out_path}")


def natural_label_key(label: str) -> List[Any]:
    parts = re.split(r"(\d+)", str(label))
    return [int(p) if p.isdigit() else p for p in parts]


def is_grouped_by_operator(rat: Any, band_type: Any) -> bool:
    """Layouts that group labels by operator (so separators between operators help)."""
    bt = str(band_type).lower()
    if bt == "mid":
        return True
    if bt == "low" and str(rat).upper() == "LTE":
        return True
    return False


def compact_band_label(band: Any, rat: str) -> str:
    return normalize_band(band, rat) or str(band).strip()


def normalize_band_type(x: Any) -> Optional[str]:
    s = str(x).strip().lower()
    if s in {"low", "mid", "mmwave"}:
        return s
    if s in {"mm wave", "millimeter", "millimeter-wave"}:
        return "mmwave"
    return None


def load_ue_record_level_report(
    report_csv: Path,
    operators: List[str],
) -> pd.DataFrame:
    if not report_csv.exists():
        print(f"[WARN] UE record report not found: {report_csv}")
        return pd.DataFrame()

    required = {
        "operator",
        "rat",
        "band",
        "band_type",
        "pcell_key",
        "rsrp",
        "interference",
        "sinr",
    }
    df = pd.read_csv(report_csv)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{report_csv} is missing required columns: {', '.join(sorted(missing))}"
        )

    op_set = {op.lower() for op in operators}
    out = pd.DataFrame(
        {
            "operator": df["operator"].astype(str).str.lower(),
            "rat": df["rat"].map(normalize_report_rat),
            "band": df.apply(
                lambda r: compact_band_label(
                    r["band"], normalize_report_rat(r["rat"]) or ""
                ),
                axis=1,
            ),
            "band_type": df["band_type"].map(normalize_band_type),
            "pcell_key": df["pcell_key"].astype(str),
            "rsrp": pd.to_numeric(df["rsrp"], errors="coerce"),
            "interference": pd.to_numeric(df["interference"], errors="coerce"),
            "sinr": pd.to_numeric(df["sinr"], errors="coerce"),
        }
    )
    out = out[
        out["operator"].isin(op_set)
        & out["rat"].isin(["NR"])
        & out["band"].notna()
        & out["band"].astype(str).str.lower().ne("unknown")
        & out["band_type"].isin(BAND_TYPE_ORDER)
        & (out["rsrp"].notna() | out["interference"].notna() | out["sinr"].notna())
    ].copy()
    out["operator_initial"] = (
        out["operator"].map(OP_INITIALS).fillna(out["operator"].str[:1].str.upper())
    )
    out["band_operator_label"] = out["band"].astype(str) + "-" + out["operator_initial"]
    return out


def load_ue_counts_for_labels(df: pd.DataFrame) -> Dict[Tuple[str, str, str], int]:
    if df.empty or "pcell_key" not in df.columns:
        return {}
    counts = (
        df.groupby(["rat", "band_type", "band_operator_label"])["pcell_key"]
        .nunique()
        .to_dict()
    )
    return {
        (str(rat), str(band_type), str(label)): int(count)
        for (rat, band_type, label), count in counts.items()
    }


def load_cell_counts_for_ue_labels(
    cell_report_csv: Path,
    operators: List[str],
) -> Dict[Tuple[str, str, str], int]:
    if not cell_report_csv.exists():
        print(
            f"[WARN] Cell report not found for UE cell-count annotations: {cell_report_csv}"
        )
        return {}

    required = {"operator", "rat", "band", "band_type", "pcell_key"}
    df = pd.read_csv(cell_report_csv)
    missing = required.difference(df.columns)
    if missing:
        print(
            f"[WARN] Cannot annotate UE cell counts; {cell_report_csv} missing "
            f"{', '.join(sorted(missing))}"
        )
        return {}

    op_set = {op.lower() for op in operators}
    work = pd.DataFrame(
        {
            "operator": df["operator"].astype(str).str.lower(),
            "rat": df["rat"].map(normalize_report_rat),
            "band": df.apply(
                lambda r: compact_band_label(
                    r["band"], normalize_report_rat(r["rat"]) or ""
                ),
                axis=1,
            ),
            "band_type": df["band_type"].map(normalize_band_type),
            "pcell_key": df["pcell_key"].astype(str),
        }
    )
    work = work[
        work["operator"].isin(op_set)
        & work["rat"].isin(["NR"])  # only NR has UE-level report for now
        & work["band"].notna()
        & work["band"].astype(str).str.lower().ne("unknown")
        & work["band_type"].isin(BAND_TYPE_ORDER)
    ].copy()
    work["operator_initial"] = (
        work["operator"].map(OP_INITIALS).fillna(work["operator"].str[:1].str.upper())
    )
    work["band_operator_label"] = (
        work["band"].astype(str) + "-" + work["operator_initial"]
    )

    counts = (
        work.groupby(["rat", "band_type", "band_operator_label"])["pcell_key"]
        .nunique()
        .to_dict()
    )
    return {
        (str(rat), str(band_type), str(label)): int(count)
        for (rat, band_type, label), count in counts.items()
    }


def operator_from_band_label(label: str) -> str:
    initial = str(label).rsplit("-", 1)[-1]
    return INITIAL_TO_OP.get(initial, initial.lower())


def ue_labels_with_min_cells(
    sub: pd.DataFrame, min_cells: int = MIN_CELLS_PER_UE_BAND
) -> set[str]:
    counts = sub.groupby("band_operator_label")["pcell_key"].nunique()
    return set(counts[counts >= int(min_cells)].index.astype(str))


def ordered_ue_band_labels(
    sub: pd.DataFrame,
    band_type: str,
    operators: List[str],
    min_cells: int = MIN_CELLS_PER_UE_BAND,
    rat: Optional[str] = None,
) -> List[str]:
    groups = (
        sub[["band", "operator_initial", "band_operator_label"]]
        .drop_duplicates()
        .copy()
    )
    kept = ue_labels_with_min_cells(sub, min_cells=min_cells)
    groups = groups[groups["band_operator_label"].astype(str).isin(kept)].copy()
    if groups.empty:
        return []
    groups["median_rsrp"] = groups["band_operator_label"].map(
        sub.groupby("band_operator_label")["rsrp"].median()
    )
    op_rank = {str(op).lower(): i for i, op in enumerate(operators)}
    groups["operator"] = groups["band_operator_label"].map(operator_from_band_label)
    groups["operator_rank"] = groups["operator"].map(
        lambda op: op_rank.get(str(op).lower(), len(op_rank))
    )
    if is_grouped_by_operator(rat, band_type):
        rows = sorted(
            groups.to_dict("records"),
            key=lambda r: (
                int(r["operator_rank"]),
                (
                    float(r["median_rsrp"])
                    if not pd.isna(r["median_rsrp"])
                    else float("inf")
                ),
                natural_label_key(str(r["band"])),
                natural_label_key(str(r["band_operator_label"])),
            ),
        )
        return [str(row["band_operator_label"]) for row in rows]

    rows = sorted(
        groups.to_dict("records"),
        key=lambda r: (
            float(r["median_rsrp"]) if not pd.isna(r["median_rsrp"]) else float("inf"),
            int(r["operator_rank"]),
            natural_label_key(str(r["band"])),
            natural_label_key(str(r["band_operator_label"])),
        ),
    )
    return [str(row["band_operator_label"]) for row in rows]


def collect_ue_box_data(
    sub: pd.DataFrame, labels: List[str], metric_col: str
) -> Tuple[List[np.ndarray], List[str]]:
    data: List[np.ndarray] = []
    kept_labels: List[str] = []
    for label in labels:
        vals = pd.to_numeric(
            sub.loc[sub["band_operator_label"] == label, metric_col],
            errors="coerce",
        ).dropna()
        if vals.empty:
            continue
        data.append(vals.to_numpy(dtype=float))
        kept_labels.append(label)
    return data, kept_labels


def mid_band_separator_positions(
    labels: List[str], band_type: str, rat: Optional[str] = None
) -> List[float]:
    if not is_grouped_by_operator(rat, band_type) or len(labels) < 2:
        return []
    positions = []
    prev_op = operator_from_band_label(labels[0])
    for idx, label in enumerate(labels[1:], start=2):
        op = operator_from_band_label(label)
        if op != prev_op:
            positions.append(idx - 0.5)
            prev_op = op
    return positions


def add_mid_band_separators(
    ax, labels: List[str], band_type: str, rat: Optional[str] = None
) -> None:
    for x in mid_band_separator_positions(labels, band_type, rat):
        ax.axvline(x=x, linestyle="--", linewidth=0.8, color="0.35", alpha=0.9)


def format_cell_count(n: Optional[int]) -> str:
    if n is None:
        return "NA cells"
    return f"{n} cell" if int(n) == 1 else f"{int(n)} cells"


def whisker_top(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanpercentile(vals, 95))


def annotate_cell_counts(
    ax,
    labels: List[str],
    y_positions: List[float],
    y_min: float,
    y_max: float,
    cell_counts: Dict[Tuple[str, str, str], int],
    rat: str,
    band_type: str,
) -> None:
    if not labels or not y_positions:
        return
    y_range = y_max - y_min if y_max > y_min else 1.0
    text_positions = []
    for idx, (label, y_base) in enumerate(zip(labels, y_positions), start=1):
        y_text = y_base + 0.04 * y_range
        text_positions.append(y_text)
        count = cell_counts.get((rat, band_type, label))
        ax.text(
            idx,
            y_text,
            format_cell_count(count),
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONT_SIZE - 2.5,
            fontweight="bold",
            rotation=0,
        )
    ax.set_ylim(
        y_min - 0.05 * y_range, max(y_max, max(text_positions)) + 0.14 * y_range
    )


def plot_ue_rsrp_interference_box(
    sub: pd.DataFrame,
    rat: str,
    band_type: str,
    out_dir: Path,
    cell_counts: Dict[Tuple[str, str, str], int],
    operators: List[str],
) -> None:
    labels = ordered_ue_band_labels(sub, band_type, operators, rat=rat)
    rsrp_data, rsrp_labels = collect_ue_box_data(sub, labels, "rsrp")
    int_data, int_labels = collect_ue_box_data(sub, labels, "interference")
    common = [
        label
        for label in labels
        if label in set(rsrp_labels) and label in set(int_labels)
    ]
    rsrp_data, labels = collect_ue_box_data(sub, common, "rsrp")
    int_data, _ = collect_ue_box_data(sub, common, "interference")
    if not rsrp_data or not int_data:
        print(f"[WARN] No UE RSRP/interference data for {rat} {band_type}")
        return

    x = np.arange(1, len(labels) + 1)
    offset = 0.18
    fig, ax = plt.subplots(figsize=pick_fig_size(rat, band_type))
    bp1 = ax.boxplot(
        rsrp_data,
        positions=x - offset,
        widths=0.3,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    bp2 = ax.boxplot(
        int_data,
        positions=x + offset,
        widths=0.3,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for box in bp1["boxes"]:
        box.set_facecolor("lightblue")
        box.set_edgecolor("black")
        box.set_hatch("////")
    for box in bp2["boxes"]:
        box.set_facecolor("salmon")
        box.set_edgecolor("black")
        box.set_hatch("\\\\\\\\")
    for median in bp1["medians"] + bp2["medians"]:
        median.set_color("black")

    ax.set_ylim(-130, -60)

    ax.set_xticks(x)
    if rat == "NR" and band_type == "mid":
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Power (dBm)")
    ax.set_xlabel("Band-Operator")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    add_mid_band_separators(ax, labels, band_type, rat)

    import matplotlib.patches as mpatches

    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor="lightblue", edgecolor="black", hatch="////", label="RSRP"
            ),
            mpatches.Patch(
                facecolor="salmon",
                edgecolor="black",
                hatch="\\\\\\\\",
                label="Interference",
            ),
        ],
        loc="best",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = out_dir / {"low": "fig_6a.pdf", "mid": "fig_6c.pdf"}[band_type]
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def plot_ue_sinr_box(
    sub: pd.DataFrame,
    rat: str,
    band_type: str,
    out_dir: Path,
    cell_counts: Dict[Tuple[str, str, str], int],
    operators: List[str],
) -> None:
    labels = ordered_ue_band_labels(sub, band_type, operators, rat=rat)
    data, labels = collect_ue_box_data(sub, labels, "sinr")
    if not data:
        print(f"[WARN] No UE SINR data for {rat} {band_type}")
        return

    fig, ax = plt.subplots(figsize=pick_fig_size(rat, band_type))
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#dbeafe")
        box.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylim(-10, 40)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    if rat == "NR" and band_type == "mid":
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("SINR (dB)")
    ax.set_xlabel("Band-Operator")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    add_mid_band_separators(ax, labels, band_type, rat)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = out_dir / {"low": "fig_6b.pdf", "mid": "fig_6d.pdf"}[band_type]
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def plot_ue_record_level_band_boxplots(
    report_csv: Path,
    cell_report_csv: Path,
    operators: List[str],
    out_dir: Path,
) -> None:
    df = load_ue_record_level_report(report_csv, operators)
    if df.empty:
        return
    cell_counts = load_ue_counts_for_labels(df)

    for rat in ["NR"]:
        for band_type in ["low", "mid"]:
            sub = df[(df["rat"] == rat) & (df["band_type"] == band_type)].copy()
            if sub.empty:
                continue
            plot_ue_rsrp_interference_box(
                sub, rat, band_type, out_dir, cell_counts, operators
            )
            plot_ue_sinr_box(sub, rat, band_type, out_dir, cell_counts, operators)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pickle-dir",
        type=str,
        default=str(ROOT / "data/intermediate_data/2024/pkl_output"),
        help="Fallback directory containing *_pcell.pkl",
    )
    ap.add_argument(
        "--cell-report",
        type=str,
        default=str(DEFAULT_CELL_REPORT),
        help="Aggregated all-cell report to use as the default input",
    )
    ap.add_argument(
        "--fig45-report",
        type=str,
        default=str(DEFAULT_FIG45_REPORT),
        help="Fallback Fig.45 cell metrics report if --cell-report is missing",
    )
    ap.add_argument(
        "--ue-record-report",
        type=str,
        default=str(DEFAULT_UE_RECORD_REPORT),
        help="UE-record-level report for per-band boxplots",
    )
    ap.add_argument(
        "--force-pickle",
        action="store_true",
        help="Ignore cached reports and recompute from *_pcell.pkl",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "plots"),
        help="Directory to write plots/csv",
    )
    ap.add_argument(
        "--operators",
        nargs="+",
        default=["att", "tmobile", "verizon"],
        help="Operators to include",
    )
    ap.add_argument(
        "--min-valid-samples",
        type=int,
        default=3,
        help="Min valid interference samples per cell",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="band_interference_per_cell.csv",
        help="CSV filename under out-dir",
    )

    # Optional city exclusion
    ap.add_argument(
        "--cities-csv",
        type=Path,
        default=None,
        help="Optional: city list with columns lat,lng",
    )
    ap.add_argument(
        "--city-radius-miles",
        type=float,
        default=10.0,
        help="City exclusion radius in miles (if cities-csv provided)",
    )

    args = ap.parse_args()
    out_dir: Path = Path(args.out_dir).resolve()
    ue_out_dir = out_dir
    cell_report: Path = Path(args.cell_report).resolve()
    ue_record_report: Path = Path(args.ue_record_report).resolve()

    ue_out_dir.mkdir(parents=True, exist_ok=True)

    plot_ue_record_level_band_boxplots(
        report_csv=ue_record_report,
        cell_report_csv=cell_report,
        operators=args.operators,
        out_dir=ue_out_dir,
    )


if __name__ == "__main__":
    main()
