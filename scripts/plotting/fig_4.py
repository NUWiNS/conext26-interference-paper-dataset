#!/usr/bin/env python3
"""LTE and 5G area-type boxplots"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PCELL_PKL_RE = re.compile(r"^(?P<op>[a-z]+)_pcell\.pkl$", re.IGNORECASE)
REPO_ROOT = Path(__file__).resolve().parents[2]

NR_BAND_COL = "5G KPI PCell RF Band"
LTE_BAND_COL = "LTE KPI PCell Serving Band"

NR_RSRP_COL = "5G KPI PCell RF Serving SS-RSRP [dBm]"
NR_SINR_COL = "5G KPI PCell RF Serving SS-SINR [dB]"

LTE_RSRP_COL = "LTE KPI PCell Serving RSRP [dBm]"
LTE_SINR_COL = "LTE KPI PCell SINR [dB]"
DEFAULT_UE_RECORD_REPORT = (
    REPO_ROOT / "data/intermediate_data/2024/complete_ue_level_metrics_report.csv"
)
DEFAULT_LTE_NEIGHBOR_CSV = REPO_ROOT / "data/intermediate_data/2024/neighbor_lte_pcell.csv"

# Global font settings for all figures
LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 15
TICK_FONTSIZE = 16
FONTWEIGHT = "bold"

OP_LABELS = {"att": "AT&T", "tmobile": "T-Mobile", "verizon": "Verizon"}
# Operator colors, matched to scripts/0_overall_interference.py
OP_COLORS = {"att": "royalblue", "tmobile": "magenta", "verizon": "red"}


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def write_stats(
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
        lines.append(f"  mean: {float(np.mean(v)):.4f} {unit}")
        for p in percentiles:
            lines.append(f"  {p}%: {float(np.percentile(v, p)):.4f} {unit}")
        lines.append(f"  max: {float(np.max(v)):.4f} {unit}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[OK] Wrote {out_path}")


def read_stats(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    label: Optional[str] = None
    for line in path.read_text().splitlines():
        m = re.match(r"^(\S.*?) \(count=(\d+)\)\s*$", line)
        if m:
            label = m.group(1)
            out[label] = {"count": float(m.group(2))}
            continue
        m = re.match(r"^\s+(min|mean|max|\d+%):\s+(-?[\d.]+)", line)
        if m and label is not None:
            out[label][m.group(1)] = float(m.group(2))
    return out


def _bxp_from_values(label: str, values: np.ndarray) -> Optional[dict]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    return {
        "label": label,
        "whislo": float(p5),
        "q1": float(p25),
        "med": float(p50),
        "q3": float(p75),
        "whishi": float(p95),
        "mean": float(np.mean(v)),
        "fliers": [],
    }


def _bxp_from_stats(label: str, d: Dict[str, float]) -> Optional[dict]:
    if not all(k in d for k in ("5%", "25%", "50%", "75%", "95%")):
        return None
    return {
        "label": label,
        "whislo": d["5%"],
        "q1": d["25%"],
        "med": d["50%"],
        "q3": d["75%"],
        "whishi": d["95%"],
        "mean": d.get("mean"),
        "fliers": [],
    }


def load_pcell_pickle(path: Path) -> dict:
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def clean_value(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def mean_numeric(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.mean()) if len(x) else float("nan")


def first_non_empty_by_scan(df: pd.DataFrame, row_indices, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    ser = df.loc[row_indices, col]
    for v in ser:
        vv = clean_value(v)
        if vv is not None:
            return vv
    return None


def infer_rat_from_key(key: str) -> Optional[str]:
    if not isinstance(key, str):
        return None
    parts = key.split("|")
    if not parts:
        return None
    rat = parts[0].strip().upper()
    return rat if rat in ("NR", "LTE") else None


def normalize_rat_label(x) -> Optional[str]:
    s = str(x).strip().upper()
    if s in {"NR", "5G"}:
        return "NR"
    if s in {"LTE", "4G"}:
        return "LTE"
    return None


def normalize_area_type(x) -> Optional[str]:
    s = str(x).strip().lower().replace("surburban", "suburban")
    if s in {"urban", "suburban", "rural"}:
        return s
    return None


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


def find_pcell_pickles(pickle_dir: Path) -> List[Path]:
    return sorted(pickle_dir.glob("*_pcell.pkl"))


def _resolve_path_with_repo_fallback(path: Path) -> Path:
    """
    Resolve user path robustly when script is launched from a different cwd.
    """
    p = Path(path)
    if p.exists():
        return p
    if p.is_absolute():
        return p
    alt = REPO_ROOT / p
    return alt if alt.exists() else p


def collect_per_cell_means_city_noncity(
    pickle_paths: List[Path],
    operators: List[str],
    min_valid_samples: int,
) -> pd.DataFrame:
    op_set = {o.lower() for o in operators}
    rows: List[Dict[str, Any]] = []
    stats = {
        "total_pcells": 0,
        "missing_or_unknown_area": 0,
        "missing_interference_col": 0,
        "low_valid_samples": 0,
    }

    for pkl in pickle_paths:
        m = PCELL_PKL_RE.match(pkl.name)
        op_from_name = m.group("op").lower() if m else ""

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
        area_map_raw = meta.get("pcell_area_type", {})
        if isinstance(area_map_raw, dict):
            area_map = {str(k): v for k, v in area_map_raw.items()}
        else:
            area_map = {}
        for key, idxs in pcell_index.items():
            stats["total_pcells"] += 1
            rat = infer_rat_from_key(key)
            if rat is None:
                continue
            inter_col = rat2inter.get(rat)
            if not inter_col or inter_col not in df.columns:
                stats["missing_interference_col"] += 1
                continue

            sub = df.loc[idxs].copy()
            inter = pd.to_numeric(sub[inter_col], errors="coerce")
            valid = inter.notna()
            if valid.sum() < min_valid_samples:
                stats["low_valid_samples"] += 1
                continue

            subv = sub.loc[valid]
            inter_v = pd.to_numeric(subv[inter_col], errors="coerce")
            band_col = NR_BAND_COL if rat == "NR" else LTE_BAND_COL
            band = first_non_empty_by_scan(df, idxs, band_col)
            rsrp_col = NR_RSRP_COL if rat == "NR" else LTE_RSRP_COL
            sinr_col = NR_SINR_COL if rat == "NR" else LTE_SINR_COL

            key_s = str(key)
            area = (
                str(area_map.get(key_s, "unknown"))
                .strip()
                .lower()
                .replace("surburban", "suburban")
            )
            if area not in {"urban", "suburban", "rural"}:
                stats["missing_or_unknown_area"] += 1
                continue

            rows.append(
                {
                    "operator": op,
                    "day": int(day) if str(day).isdigit() else day,
                    "rat": rat,
                    "band": band,
                    "pcell_key": key,
                    "area_type": area,
                    "cell_mean_interference": float(inter_v.mean()),
                    "cell_mean_rsrp": (
                        mean_numeric(sub[rsrp_col])
                        if rsrp_col in df.columns
                        else float("nan")
                    ),
                    "cell_mean_sinr": (
                        mean_numeric(sub[sinr_col])
                        if sinr_col in df.columns
                        else float("nan")
                    ),
                    "n_valid_samples": int(len(inter_v)),
                    "n_area_labeled_samples": int(len(inter_v)),
                    "area_fraction": 1.0,
                    "pcell_pickle": pkl.name,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.attrs["collect_stats"] = stats
    return out_df


def load_ue_record_area_type(report_csv: Path, operators: List[str]) -> pd.DataFrame:
    """
    Load UE-perspective record-level metrics with area labels.
    Expected source: data/intermediate_data/2024/complete_ue_level_metrics_report.csv
    """
    if not report_csv.exists():
        raise SystemExit(f"UE record metrics report not found: {report_csv}")

    required_cols = [
        "operator",
        "rat",
        "area_type",
        "interference",
        "rsrp",
        "sinr",
        "neighbor_count",
    ]
    try:
        df = pd.read_csv(report_csv, usecols=required_cols)
    except ValueError as exc:
        header = pd.read_csv(report_csv, nrows=0).columns.tolist()
        missing = sorted(set(required_cols).difference(header))
        raise SystemExit(
            f"UE record metrics report is missing columns {missing}: {report_csv}"
        ) from exc

    op_set = {str(o).strip().lower() for o in operators}
    out = pd.DataFrame(
        {
            "operator": df["operator"].astype(str).str.strip().str.lower(),
            "rat": df["rat"].map(normalize_rat_label),
            "area_type": df["area_type"].map(normalize_area_type),
            "ue_interference": pd.to_numeric(df["interference"], errors="coerce"),
            "ue_rsrp": pd.to_numeric(df["rsrp"], errors="coerce"),
            "ue_sinr": pd.to_numeric(df["sinr"], errors="coerce"),
            "ue_neighbor_count": pd.to_numeric(df["neighbor_count"], errors="coerce"),
        }
    )
    out = out[
        out["operator"].isin(op_set)
        & out["rat"].isin(["NR", "LTE"])
        & out["area_type"].isin(["urban", "suburban", "rural"])
        & out[["ue_interference", "ue_rsrp", "ue_sinr", "ue_neighbor_count"]]
        .notna()
        .any(axis=1)
    ].copy()
    print(f"[OK] Loaded {report_csv} ({len(out)} UE records with area_type)")
    return out


def load_lte_neighbor_count_by_area(
    neighbor_csv: Path,
    operators: List[str],
) -> pd.DataFrame:
    if not neighbor_csv.exists():
        raise SystemExit(f"LTE neighbor source not found: {neighbor_csv}")

    op_set = {str(o).strip().lower() for o in operators}

    n = pd.read_csv(
        neighbor_csv,
        low_memory=False,
        usecols=[
            "source_file",
            "operator",
            "time",
            "channel_key",
            "neighbor_count",
            "area",
        ],
    )
    if "area" not in n.columns:
        raise SystemExit(
            f"{neighbor_csv} missing 'area' column; regenerate it with "
            "scripts/data_cleaning/parse_2024_lte_neighbors.py before plotting."
        )
    n["operator"] = n["operator"].astype(str).str.strip().str.lower()
    n["neighbor_count"] = pd.to_numeric(n["neighbor_count"], errors="coerce")
    n["channel_key"] = pd.to_numeric(n["channel_key"], errors="coerce")
    n["time"] = pd.to_datetime(n["time"], errors="coerce")
    n["area"] = n["area"].map(normalize_area_type)
    n = n[
        n["operator"].isin(op_set)
        & n["neighbor_count"].notna()
        & n["channel_key"].notna()
        & n["time"].notna()
        & n["area"].isin(["urban", "suburban", "rural"])
    ].copy()
    n["t_bucket"] = n["time"].astype("int64") // 100_000_000  # 100-ms buckets

    bucket_keys = ["operator", "source_file", "channel_key", "t_bucket"]
    n_grp = n.groupby(bucket_keys, as_index=False)["neighbor_count"].max()
    area_counts = n.groupby(bucket_keys + ["area"], as_index=False).size()
    # Match Series.mode() semantics: highest frequency, then lexical first on ties.
    area_mode = (
        area_counts.sort_values(["size", "area"], ascending=[False, True], kind="stable")
        .drop_duplicates(bucket_keys)[bucket_keys + ["area"]]
    )
    n_grp = n_grp.merge(area_mode, on=bucket_keys, how="left", validate="one_to_one")
    print(
        f"[INFO] LTE neighbor: {len(n):,} raw rows -> {len(n_grp):,} buckets "
        f"after per-channel 100-ms max-N (area from data/raw_data/2024/area.csv asof-merge)"
    )

    return pd.DataFrame(
        {
            "operator": n_grp["operator"].astype(str),
            "rat": "LTE",
            "area_type": n_grp["area"],
            "ue_neighbor_count": n_grp["neighbor_count"].astype(float),
        }
    )


def plot_cdf_city_noncity(
    df_cells: pd.DataFrame,
    rat: str,
    operators: List[str],
    metric_col: str,
    x_label: str,
    title_metric: str,
    out_path: Path,
    title_prefix: str = "PCell mean",
) -> None:
    sub = df_cells[df_cells["rat"].astype(str).str.upper() == rat.upper()]
    if sub.empty:
        print(f"[WARN] No cells for RAT={rat}")
        return

    # Color by operator
    op_color = {
        "att": "tab:blue",
        "tmobi": "tab:orange",
        "tmobile": "tab:orange",
        "verizon": "tab:green",
    }
    # Line style by area type
    area_ls = {
        "rural": "-",
        "suburban": "-.",
        "urban": ":",
    }

    plt.figure()
    for op in operators:
        op_key = op.lower()
        color = op_color.get(op_key, None)

        for area in ["rural", "suburban", "urban"]:
            ss = sub[
                (sub["operator"].astype(str).str.lower() == op_key)
                & (sub["area_type"].astype(str).str.lower() == area)
            ][metric_col].to_numpy(dtype=float)

            xs, ys = ecdf(ss)
            if xs.size == 0:
                print(f"[WARN] No data for {rat} {op} {area} {metric_col}")
                continue

            plt.plot(
                xs,
                ys,
                label=f"{op}-{area}",
                linestyle=area_ls[area],
                color=color,
            )

    plt.xlabel(x_label)
    plt.ylabel("CDF")
    plt.title(f"{rat} - Area-type {title_prefix} {title_metric}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Wrote {out_path}")


def plot_box_city_noncity(
    df_cells: Optional[pd.DataFrame],
    rat: str,
    operators: List[str],
    metric_col: str,
    y_label: str,
    title_metric: str,
    out_path: Path,
    title_prefix: str = "PCell mean",
    from_stats: bool = False,
) -> None:
    area_order = ["rural", "suburban", "urban"]

    bxp_stats: List[dict] = []
    area_labels: List[str] = []
    box_ops: List[str] = []  # operator key per box, for coloring
    op_groups: List[Tuple[str, List[int]]] = []  # (op_label, box positions)
    stats_series: List[Tuple[str, np.ndarray]] = []  # for write_stats (data mode)
    pos = 0

    if from_stats:
        stats_path = out_path.with_suffix(".txt")
        if not stats_path.exists():
            print(f"[WARN] No stats file to plot from: {stats_path}")
            return
        parsed = read_stats(stats_path)
        for op in operators:
            op_key = op.lower()
            op_label = OP_LABELS.get(op_key, op.upper())
            positions: List[int] = []
            for area in area_order:
                box = _bxp_from_stats(
                    f"{op_label}-{area}", parsed.get(f"{op_label}-{area}", {})
                )
                if box is None:
                    continue
                pos += 1
                bxp_stats.append(box)
                area_labels.append(area)
                box_ops.append(op_key)
                positions.append(pos)
            if positions:
                op_groups.append((op_label, positions))
    else:
        sub = df_cells[df_cells["rat"].astype(str).str.upper() == rat.upper()].copy()
        if sub.empty:
            print(f"[WARN] No cells for RAT={rat}")
            return
        for op in operators:
            op_key = op.lower()
            op_label = OP_LABELS.get(op_key, op.upper())
            positions = []
            for area in area_order:
                ss = sub.loc[
                    (sub["operator"].astype(str).str.lower() == op_key)
                    & (sub["area_type"].astype(str).str.lower() == area),
                    metric_col,
                ].dropna()
                if ss.empty:
                    print(f"[WARN] No box data for {rat} {op} {area} {metric_col}")
                    continue
                vals = ss.to_numpy(dtype=float)
                box = _bxp_from_values(f"{op_label}-{area}", vals)
                if box is None:
                    continue
                pos += 1
                bxp_stats.append(box)
                area_labels.append(area)
                box_ops.append(op_key)
                positions.append(pos)
                stats_series.append((f"{op_label}-{area}", vals))
            if positions:
                op_groups.append((op_label, positions))

    if not bxp_stats:
        print(f"[WARN] No box data for {rat} {metric_col}")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))
    positions = list(range(1, len(bxp_stats) + 1))
    bp = ax.bxp(bxp_stats, positions=positions, showfliers=False, patch_artist=True)
    for patch, op_key in zip(bp["boxes"], box_ops):
        patch.set_facecolor(OP_COLORS.get(op_key, "tab:gray"))
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("black")

    # Vertical dashed lines separating operator groups (skip the right edge)
    for _, grp in op_groups[:-1]:
        ax.axvline(
            x=grp[-1] + 0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7
        )

    # Lower tier: area type directly under each box (abbreviated R/S/U)
    area_abbr = {"rural": "R", "suburban": "S", "urban": "U"}
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [area_abbr.get(a, a) for a in area_labels],
        ha="center",
        fontsize=TICK_FONTSIZE,
        fontweight=FONTWEIGHT,
    )
    # Upper tier: one operator name centered under each group of area boxes
    for op_label, grp in op_groups:
        ax.text(
            sum(grp) / len(grp),
            -0.10,
            op_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=LABEL_FONTSIZE,
            fontweight=FONTWEIGHT,
            clip_on=False,
        )

    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight(FONTWEIGHT)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")

    if not from_stats and out_path.suffix.lower() == ".png":
        write_stats(
            out_path.with_suffix(".txt"),
            title=f"{rat} - Area-type {title_prefix} {title_metric} Statistics",
            unit=y_label,
            series=stats_series,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pickle-dir",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/pkl_output",
        help="Directory containing *_pcell.pkl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plots",
        help="Output directory",
    )
    ap.add_argument(
        "--operators",
        nargs="+",
        default=["att", "tmobile", "verizon"],
        # default=["att"],
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
        default="area_type_per_cell.csv",
        help="CSV filename under out-dir",
    )
    ap.add_argument(
        "--ue-record-report",
        type=Path,
        default=DEFAULT_UE_RECORD_REPORT,
        help=(
            "UE record metrics report with operator, rat, area_type, "
            "interference, rsrp, sinr, and neighbor_count columns"
        ),
    )
    ap.add_argument(
        "--lte-neighbor-csv",
        type=Path,
        default=DEFAULT_LTE_NEIGHBOR_CSV,
        help=(
            "Raw-log-parsed LTE neighbor CSV (data/intermediate_data/2024/neighbor_lte_pcell.csv); "
            "must contain the 'area' column generated from data/raw_data/2024/area.csv"
        ),
    )
    ap.add_argument(
        "--from-stats",
        action="store_true",
        help=(
            "Redraw boxes from the existing *_box.txt statistics instead of "
            "reloading the raw pickles/CSVs (fast when only changing formatting)"
        ),
    )
    ap.add_argument(
        "--build-cell-summary",
        action="store_true",
        help="Also build or load the legacy per-cell summary (not used by the UE-level paper panels)",
    )

    args = ap.parse_args()
    args.pickle_dir = _resolve_path_with_repo_fallback(args.pickle_dir)
    args.ue_record_report = _resolve_path_with_repo_fallback(args.ue_record_report)
    args.lte_neighbor_csv = _resolve_path_with_repo_fallback(args.lte_neighbor_csv)
    args.out_dir = Path(args.out_dir)

    if args.from_stats:
        print("[INFO] --from-stats: redrawing boxes from existing *_box.txt files")

    if args.build_cell_summary and not args.from_stats:
        out_csv_path = args.out_dir / args.out_csv
        required_cols = {"band", "cell_mean_rsrp", "cell_mean_sinr"}
        should_recompute = not out_csv_path.exists()
        if not should_recompute:
            existing_df = pd.read_csv(out_csv_path)
            missing_cols = required_cols.difference(existing_df.columns)
            if missing_cols:
                print(
                    f"[INFO] Recomputing {out_csv_path} to add missing columns: "
                    f"{sorted(missing_cols)}"
                )
                should_recompute = True
            else:
                df_cells = existing_df
                print(f"[OK] Loaded {out_csv_path} ({len(df_cells)} cells)")

        if should_recompute:
            args.out_dir.mkdir(parents=True, exist_ok=True)

            pkls = find_pcell_pickles(args.pickle_dir)
            if not pkls:
                raise SystemExit(f"No *_pcell.pkl found in {args.pickle_dir}")

            df_cells = collect_per_cell_means_city_noncity(
                pickle_paths=pkls,
                operators=args.operators,
                min_valid_samples=int(args.min_valid_samples),
            )
            if df_cells.empty:
                st = df_cells.attrs.get("collect_stats", {})
                raise SystemExit(
                    "No cells found with valid metadata area_type and interference. "
                    f"stats={st}"
                )

            df_cells.to_csv(out_csv_path, index=False)
            print(f"[OK] Wrote {out_csv_path} ({len(df_cells)} cells)")

    plot_specs = [
        (
            "cell_mean_interference",
            "Per-cell mean Interference [dBm] (RSRP - SINR)",
            "Interference",
            "cdf",
        ),
        (
            "cell_mean_rsrp",
            "Per-cell mean RSRP [dBm]",
            "RSRP",
            "rsrp_cdf",
        ),
        (
            "cell_mean_sinr",
            "Per-cell mean SINR [dB]",
            "SINR",
            "sinr_cdf",
        ),
    ]
    # for rat in ["NR", "LTE"]:
    #     for metric_col, x_label, title_metric, file_stem in plot_specs:
    #         plot_cdf_city_noncity(
    #             df_cells,
    #             rat,
    #             args.operators,
    #             metric_col,
    #             x_label,
    #             title_metric,
    #             args.out_dir / f"area_type_{rat}_{file_stem}.png",
    #         )
    #         plot_box_city_noncity(
    #             df_cells,
    #             rat,
    #             args.operators,
    #             metric_col,
    #             x_label,
    #             title_metric,
    #             args.out_dir / f"area_type_{rat}_{file_stem.replace('cdf', 'box')}.png",
    #         )

    df_ue: Optional[pd.DataFrame] = None
    df_lte_neighbor: Optional[pd.DataFrame] = None
    if not args.from_stats:
        df_ue = load_ue_record_area_type(args.ue_record_report, args.operators)
        if df_ue.empty:
            raise SystemExit(
                "No UE records found with valid operator, RAT, area_type, and metrics "
                f"in {args.ue_record_report}"
            )

        df_lte_neighbor = load_lte_neighbor_count_by_area(
            args.lte_neighbor_csv, args.operators
        )

    ue_out_dir = args.out_dir
    ue_plot_specs = [
        (
            "ue_interference",
            "Interference (dBm)",
            "Interference",
            "interference_cdf",
        ),
        (
            "ue_rsrp",
            "RSRP (dBm)",
            "RSRP",
            "rsrp_cdf",
        ),
        (
            "ue_sinr",
            "SINR (dB)",
            "SINR",
            "sinr_cdf",
        ),
        (
            "ue_neighbor_count",
            "#Interfering Cells",
            "Number of Interferers",
            "neighbor_count_cdf",
        ),
    ]
    panel_names = {
        ("LTE", "ue_rsrp"): "fig_4a.pdf",
        ("LTE", "ue_interference"): "fig_4b.pdf",
        ("LTE", "ue_sinr"): "fig_4c.pdf",
        ("LTE", "ue_neighbor_count"): "fig_4d.pdf",
        ("NR", "ue_rsrp"): "fig_4e.pdf",
        ("NR", "ue_interference"): "fig_4f.pdf",
        ("NR", "ue_sinr"): "fig_4g.pdf",
        ("NR", "ue_neighbor_count"): "fig_4h.pdf",
    }
    for rat in ["NR", "LTE"]:
        for metric_col, x_label, title_metric, file_stem in ue_plot_specs:
            source_df = df_ue
            if rat == "LTE" and metric_col == "ue_neighbor_count":
                source_df = df_lte_neighbor
            # plot_cdf_city_noncity(
            #     source_df,
            #     rat,
            #     args.operators,
            #     metric_col,
            #     x_label,
            #     title_metric,
            #     ue_out_dir / f"area_type_{rat}_{file_stem}.png",
            #     title_prefix="UE record",
            # )
            plot_box_city_noncity(
                source_df,
                rat,
                args.operators,
                metric_col,
                x_label,
                title_metric,
                ue_out_dir / panel_names[(rat, metric_col)],
                title_prefix="UE record",
                from_stats=args.from_stats,
            )

    # n77 AT&T-vs-Verizon area-type comparison: 2 figs (RSRP, interference),
    # each with 6 boxes (2 operators grouped within each area type).
    # n77_ops = [o for o in args.operators if o.lower() in {"att", "verizon"}]
    # if len(n77_ops) >= 2:
    #     df_n77 = load_n77_area_records(args.ue_record_report, tuple(n77_ops))
    #     if df_n77.empty:
    #         print("[WARN] No n77 records for AT&T/Verizon; skipping n77 area plots")
    #     else:
    #         n77_specs = [
    #             ("rsrp", "RSRP (dBm)", "RSRP"),
    #             ("interference", "Interference (dBm)", "Interference"),
    #         ]
    #         for metric_col, y_label, title_metric in n77_specs:
    #             for ext in ("png", "pdf"):
    #                 plot_box_n77_area_grouped(
    #                     df_n77,
    #                     n77_ops,
    #                     metric_col,
    #                     y_label,
    #                     title_metric,
    #                     ue_out_dir / f"area_type_n77_att_vz_{metric_col}_box.{ext}",
    #                 )
    # else:
    #     print("[INFO] Need both att and verizon in --operators for n77 area plots")


if __name__ == "__main__":
    main()
