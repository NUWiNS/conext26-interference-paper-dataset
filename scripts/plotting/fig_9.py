#!/usr/bin/env python3
"""Interfering cell evolution CDFs"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Global font and figure-size settings for all figures.
LABEL_FONT_SIZE = 17
AXIS_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15
FIG_SIZE = (6, 4)

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

# Show the legend only on the AT&T figure.
LEGEND_OP = "att"

OPS = ["att", "tmobile", "verizon"]
OP_LABELS = {"att": "AT&T", "tmobile": "T-Mobile", "verizon": "Verizon"}
OP_COLORS = {"att": "royalblue", "tmobile": "magenta", "verizon": "red"}

RAT_COLORS = {"LTE": "#2ca02c", "5G": "#ff160e"}
YEAR_STYLES = {
    "2022": {"linestyle": "--", "linewidth": 2.2},
    "2024": {"linestyle": "-", "linewidth": 2.5},
}
YEAR_HATCHES = {"2022": None, "2024": "////"}
RAT_ORDER = ["LTE", "5G"]
YEAR_ORDER = ["2022", "2024"]

OPERATOR_TO_SHORT = {
    "AT&T": "att",
    "T-Mobile": "tmobile",
    "Verizon": "verizon",
    "att": "att",
    "tmobile": "tmobile",
    "verizon": "verizon",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_lte(path: Path, operators: List[str], tag: str) -> pd.DataFrame:
    """LTE: parsed-neighbor CSV with per-channel 100-ms max-N bucketing."""
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=["source_file", "operator", "time", "channel_key", "neighbor_count"],
    )
    df["operator"] = df["operator"].map(
        lambda v: OPERATOR_TO_SHORT.get(str(v), str(v).lower())
    )
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")
    df = df[df["operator"].isin(operators) & df["neighbor_count"].notna()].copy()
    n_raw = len(df)

    df["time"] = pd.to_datetime(df["time"])
    df["t_bucket_ns"] = df["time"].astype("int64") // 100_000_000
    df = df.groupby(
        ["source_file", "channel_key", "t_bucket_ns", "operator"], as_index=False
    )["neighbor_count"].max()
    print(
        f"[INFO] {tag} LTE: rows {n_raw:,} -> {len(df):,} after 100-ms per-channel max-N"
    )
    return df[["operator", "neighbor_count"]]


def load_5g_2022(path: Path, operators: List[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path, low_memory=False, usecols=["operator", "rat", "neighbor_count"]
    )
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].astype(str).str.upper().replace({"NR": "5G", "LTE": "4G"})
    df = df[df["rat"].eq("5G") & df["operator"].isin(operators)].copy()
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce").fillna(
        0
    )
    print(f"[INFO] 2022 5G: rows {len(df):,} (NR-only; NaN counted as 0)")
    return df[["operator", "neighbor_count"]]


def load_5g_2024(path: Path, operators: List[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path, low_memory=False, usecols=["operator", "rat", "neighbor_count"]
    )
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].astype(str).str.upper().replace({"NR": "5G", "LTE": "4G"})
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")
    df = df[
        df["rat"].eq("5G")
        & df["operator"].isin(operators)
        & df["neighbor_count"].notna()
    ].copy()
    print(f"[INFO] 2024 5G: {len(df):,} NR rows with neighbor_count")
    return df[["operator", "neighbor_count"]]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def build_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    v.sort()
    return v, np.arange(1, v.size + 1) / v.size


def _save(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Wrote {out_path}")


def values_for(sources, op: str, rat: str, year: str) -> np.ndarray:
    df = sources[(rat, year)]
    return df.loc[df["operator"] == op, "neighbor_count"].to_numpy(dtype=float)


def plot_op_cdf(op: str, sources, out_path: Path) -> None:
    plt.figure(figsize=FIG_SIZE)
    plotted = False
    for rat in RAT_ORDER:
        color = RAT_COLORS[rat]
        for year in YEAR_ORDER:
            vals = values_for(sources, op, rat, year)
            x, y = build_cdf(vals)
            if x.size == 0:
                continue
            plt.plot(
                x,
                y,
                color=color,
                label=f"{rat} {year}",
                **YEAR_STYLES[year],
            )
            plotted = True
    if not plotted:
        plt.close()
        return

    plt.xlabel("#Interfering Cells")
    plt.ylabel("CDF")
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if op == LEGEND_OP:
        plt.legend()
    _save(out_path)
    plt.close()


def plot_op_box(op: str, sources, out_path: Path) -> None:
    pair_gap = 1.0
    group_gap = 2.0
    width = 0.7

    positions, data, colors, hatches = [], [], [], []
    rat_centers = []
    separators = []

    cursor = 1.0
    for rat in RAT_ORDER:
        rat_color = RAT_COLORS[rat]
        first_pos = cursor
        for year in YEAR_ORDER:
            vals = values_for(sources, op, rat, year)
            if vals.size == 0:
                cursor += pair_gap
                continue
            positions.append(cursor)
            data.append(vals)
            colors.append(rat_color)
            hatches.append(YEAR_HATCHES[year])
            cursor += pair_gap
        last_pos = cursor - pair_gap
        rat_centers.append((first_pos + last_pos) / 2.0)
        cursor += group_gap - pair_gap
        separators.append(cursor - (group_gap - pair_gap) / 2.0)

    if not data:
        print(f"[WARN] no interferer data for {op}")
        return
    separators = separators[:-1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        showfliers=False,
        patch_artist=True,
        whis=(5, 95),
    )
    for box, c, h in zip(bp["boxes"], colors, hatches):
        box.set_facecolor(c)
        box.set_edgecolor("black")
        if h:
            box.set_hatch(h)
    for median in bp["medians"]:
        median.set_color("black")

    for x in separators:
        ax.axvline(x=x, linestyle="--", linewidth=0.8, color="0.35", alpha=0.9)

    ax.set_xticks(rat_centers)
    ax.set_xticklabels(RAT_ORDER)
    ax.set_xlim(0.0, cursor - group_gap + 1.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("#Interfering Cells")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    if op == LEGEND_OP:
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="lightgray", edgecolor="black", label="2022"),
                mpatches.Patch(
                    facecolor="lightgray", edgecolor="black", hatch="////", label="2024"
                ),
            ],
            loc="best",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Wrote {out_path}")
    plt.close(fig)


def compute_summary(sources, operators: List[str]) -> pd.DataFrame:
    rows = []
    for year in YEAR_ORDER:
        for rat in RAT_ORDER:
            df = sources[(rat, year)]
            for op in operators:
                vals = df.loc[df["operator"] == op, "neighbor_count"].to_numpy(
                    dtype=float
                )
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    rows.append({"year": year, "rat": rat, "operator": op, "count": 0})
                    continue
                p5, q1, median, q3, p95 = np.nanpercentile(vals, [5, 25, 50, 75, 95])
                rows.append(
                    {
                        "year": year,
                        "rat": rat,
                        "operator": op,
                        "count": int(vals.size),
                        "mean": float(np.nanmean(vals)),
                        "std": (
                            float(np.nanstd(vals, ddof=1))
                            if vals.size > 1
                            else float("nan")
                        ),
                        "min": float(np.nanmin(vals)),
                        "p5": float(p5),
                        "q1": float(q1),
                        "median": float(median),
                        "q3": float(q3),
                        "p95": float(p95),
                        "max": float(np.nanmax(vals)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lte-2022",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2022/neighbor_lte_pcell.csv",
    )
    ap.add_argument(
        "--lte-2024",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/neighbor_lte_pcell.csv",
    )
    ap.add_argument(
        "--nr-2022",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2022/per_cell_all.csv",
    )
    ap.add_argument(
        "--nr-2024",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/complete_ue_level_metrics_report.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plots",
    )
    ap.add_argument(
        "--summary-csv", type=Path,
        default=REPO_ROOT / "data/intermediate_data/2022/fig_9_summary.csv",
    )
    ap.add_argument("--operators", nargs="+", default=OPS)
    args = ap.parse_args()

    operators = [str(o).lower() for o in args.operators if str(o).lower() in OPS]
    if not operators:
        raise SystemExit("No supported operators requested")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        ("LTE", "2022"): load_lte(args.lte_2022, operators, tag="2022"),
        ("LTE", "2024"): load_lte(args.lte_2024, operators, tag="2024"),
        ("5G", "2022"): load_5g_2022(args.nr_2022, operators),
        ("5G", "2024"): load_5g_2024(args.nr_2024, operators),
    }

    summary = compute_summary(sources, operators)
    summary_csv = args.summary_csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote {summary_csv}")

    for letter, op in (("a", "att"), ("b", "tmobile"), ("c", "verizon")):
        if op in operators:
            plot_op_cdf(op, sources, args.out_dir / f"fig_9{letter}.pdf")


if __name__ == "__main__":
    main()
