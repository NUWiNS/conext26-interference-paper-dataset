#!/usr/bin/env python3
"""2024 neighboring cell CDFs"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

OPS = ["att", "tmobile", "verizon"]
OP_LABELS = {"att": "AT&T", "tmobile": "T-Mobile", "verizon": "Verizon"}
OP_COLORS = {"att": "royalblue", "tmobile": "magenta", "verizon": "red"}

# Global font settings for all CDF figures
LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 11
TICK_FONTSIZE = 12
FONTWEIGHT = "bold"
FIG_SIZE = (4, 3)


def build_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    v.sort()
    return v, np.arange(1, v.size + 1) / v.size


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_lte_from_parsed_neighbors(path: Path, operators: List[str]) -> pd.DataFrame:
    """LTE: parsed-neighbor CSV + per-channel 100-ms max-N bucket."""
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=["source_file", "operator", "time", "channel_key", "neighbor_count"],
    )
    df["operator"] = df["operator"].astype(str).str.lower()
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")
    df = df[df["operator"].isin(operators) & df["neighbor_count"].notna()].copy()
    n_raw = len(df)

    df["time"] = pd.to_datetime(df["time"])
    df["t_bucket_ns"] = df["time"].astype("int64") // 100_000_000
    df = df.groupby(
        ["source_file", "channel_key", "t_bucket_ns", "operator"], as_index=False
    )["neighbor_count"].max()
    print(
        f"[INFO] LTE: rows {n_raw:,} -> {len(df):,} after 100-ms per-channel max-N bucket"
    )
    return df[["operator", "neighbor_count"]]


def load_5g_from_ue_report(path: Path, operators: List[str]) -> pd.DataFrame:
    """5G: NR rows from complete_ue_level_metrics_report.csv."""
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=["operator", "rat", "neighbor_count"],
    )
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].astype(str).str.upper().replace({"NR": "5G", "LTE": "4G"})
    df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")
    df = df[
        df["rat"].eq("5G")
        & df["operator"].isin(operators)
        & df["neighbor_count"].notna()
    ].copy()
    print(f"[INFO] 5G: {len(df):,} NR rows with neighbor_count")
    return df[["operator", "neighbor_count"]]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_cdf(
    df: pd.DataFrame,
    operators: List[str],
    title: str,
    out_path_stem: Path,
    show_legend: bool = False,
) -> None:
    plt.figure(figsize=FIG_SIZE)
    plotted = False
    for op in operators:
        vals = df.loc[df["operator"] == op, "neighbor_count"].to_numpy(dtype=float)
        x, y = build_cdf(vals)
        if x.size == 0:
            continue
        plt.plot(
            x,
            y,
            label=OP_LABELS.get(op, op.upper()),
            color=OP_COLORS.get(op),
            linewidth=2.5,
        )
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("#Interfering Cells", fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.ylabel("CDF", fontsize=LABEL_FONTSIZE, fontweight=FONTWEIGHT)
    plt.xticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    plt.yticks(fontsize=TICK_FONTSIZE, fontweight=FONTWEIGHT)
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if show_legend:
        plt.legend(prop={"size": LEGEND_FONTSIZE, "weight": FONTWEIGHT})
    plt.tight_layout()
    p = out_path_stem.with_suffix(".pdf")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print(f"[OK] Wrote {p}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lte-csv",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/neighbor_lte_pcell.csv",
    )
    ap.add_argument(
        "--ue-report",
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
        default=REPO_ROOT / "data/intermediate_data/2024/fig_3_summary.csv",
    )
    ap.add_argument("--operators", nargs="+", default=OPS)
    args = ap.parse_args()

    operators = [str(op).lower() for op in args.operators if str(op).lower() in OPS]
    if not operators:
        raise SystemExit("No supported operators requested")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading LTE source: {args.lte_csv}")
    lte = load_lte_from_parsed_neighbors(args.lte_csv, operators)
    print(f"[INFO] loading 5G source:  {args.ue_report}")
    nr = load_5g_from_ue_report(args.ue_report, operators)

    # Summary CSV with both RATs
    rows = []
    for rat_lbl, sub in [("LTE", lte), ("5G", nr)]:
        if sub.empty:
            continue
        g = (
            sub.groupby("operator")["neighbor_count"]
            .agg(
                samples="size",
                median="median",
                mean="mean",
                p25=lambda s: np.percentile(s, 25),
                p75=lambda s: np.percentile(s, 75),
                max="max",
            )
            .reset_index()
        )
        g.insert(1, "rat", rat_lbl)
        rows.append(g)
    summary = pd.concat(rows, ignore_index=True)
    summary_csv = args.summary_csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"\n[OK] Wrote {summary_csv}")
    print(summary.to_string(index=False))

    print()
    plot_cdf(
        lte,
        operators,
        "LTE UE Perspective Interferers CDF",
        args.out_dir / "fig_3a",
        show_legend=True,
    )
    plot_cdf(
        nr,
        operators,
        "5G UE Perspective Interferers CDF",
        args.out_dir / "fig_3b",
        show_legend=False,
    )


if __name__ == "__main__":
    main()
