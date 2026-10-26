#!/usr/bin/env python3
"""RF-quality evolution boxplots"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

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

LEGEND_FIGURE = "rsrp"
LEGEND_OP = "att"

OPS = ["att", "tmobile", "verizon"]
OP_LABELS = {"att": "AT&T", "tmobile": "T-Mobile", "verizon": "Verizon"}
OP_COLORS = {"att": "royalblue", "tmobile": "magenta", "verizon": "red"}

RAT_ORDER = ["LTE", "NR"]

YEAR_STYLES = {
    "2022": {"linestyle": "--", "linewidth": 2.2},
    "2024": {"linestyle": "-", "linewidth": 2.5},
}

METRICS = {
    "interference": {
        "label": "Interference",
        "xlabel": "PCell Interference (dBm)",
        "xlim": (-130, -70),
        "col_2022": "interference",
        "col_2024": "interference_dbm",
    },
    "rsrp": {
        "label": "RSRP",
        "xlabel": "PCell RSRP (dBm)",
        "xlim": (-120, -50),
        "col_2022": "rsrp",
        "col_2024": "rsrp_dbm",
    },
    "sinr": {
        "label": "SINR",
        "xlabel": "PCell SINR (dB)",
        "xlim": (-10, 40),
        "col_2022": "sinr",
        "col_2024": "sinr_db",
    },
}


def normalize_rat(x) -> str:
    s = str(x).strip().upper()
    if s in {"NR", "5G"}:
        return "NR"
    if s in {"LTE", "4G"}:
        return "LTE"
    return ""


def load_2022(csv_path: Path) -> pd.DataFrame:
    use = ["operator", "rat", "interference", "rsrp", "sinr"]
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in use)
    missing = [c for c in use if c not in df.columns]
    if missing:
        raise SystemExit(f"2022 per_cell_all.csv missing columns: {missing}")
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].map(normalize_rat)
    for c in ("interference", "rsrp", "sinr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_2024(cache_dir: Path, operators: List[str]) -> pd.DataFrame:
    use = ["operator", "rat", "interference_dbm", "rsrp_dbm", "sinr_db"]
    parts = []
    for op in operators:
        for rat in RAT_ORDER:
            pattern = f"overall_interference_{op}_{rat.lower()}*.csv"
            files = sorted(cache_dir.glob(pattern))
            for f in files:
                try:
                    parts.append(
                        pd.read_csv(f, low_memory=False, usecols=lambda c: c in use)
                    )
                except Exception as exc:
                    print(f"[WARN] failed to read {f}: {exc}")
    if not parts:
        raise SystemExit(
            f"No 2024 cache CSVs found under {cache_dir}. "
            f"Run scripts/plotting/fig_2_5_15.py first."
        )
    df = pd.concat(parts, ignore_index=True)
    df["operator"] = df["operator"].astype(str).str.lower()
    df["rat"] = df["rat"].map(normalize_rat)
    for c in ("interference_dbm", "rsrp_dbm", "sinr_db"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_cdf(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    v.sort()
    return v, np.arange(1, v.size + 1) / v.size


def _save(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Wrote {out_path}")


def plot_all_ops_rat_metric(
    rat: str,
    metric_name: str,
    df_2022: pd.DataFrame,
    df_2024: pd.DataFrame,
    operators: List[str],
    out_path: Path,
) -> None:
    """One figure per (rat, metric); every operator × year overlaid."""
    metric = METRICS[metric_name]

    plt.figure(figsize=FIG_SIZE)
    plotted = False
    for op in operators:
        color = OP_COLORS.get(op, "black")
        for year, df, col in (
            ("2022", df_2022, metric["col_2022"]),
            ("2024", df_2024, metric["col_2024"]),
        ):
            vals = (
                df.loc[(df["operator"] == op) & (df["rat"] == rat), col]
                .dropna()
                .to_numpy(dtype=float)
            )
            xs, ys = build_cdf(vals)
            if xs.size == 0:
                continue
            plt.plot(
                xs,
                ys,
                color=color,
                label=f"{OP_LABELS.get(op, op.upper())} {year}",
                **YEAR_STYLES[year],
            )
            plotted = True

    if not plotted:
        print(f"[WARN] no data for all-ops {rat} {metric_name}")
        plt.close()
        return

    plt.xlabel(metric["xlabel"])
    plt.ylabel("CDF")
    plt.xlim(*metric["xlim"])
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if metric_name == LEGEND_FIGURE:
        plt.legend()
    _save(out_path)
    plt.close()


def plot_all_ops_rat_metric_box(
    rat: str,
    metric_name: str,
    df_2022: pd.DataFrame,
    df_2024: pd.DataFrame,
    operators: List[str],
    out_path: Path,
) -> None:
    """Boxplot version: one figure per (rat, metric); 3 ops x 2 years = 6 boxes.
    """
    metric = METRICS[metric_name]

    pair_gap = 1.0
    group_gap = 2.0
    width = 0.7

    positions_22: List[float] = []
    positions_24: List[float] = []
    data_22: List[np.ndarray] = []
    data_24: List[np.ndarray] = []
    colors_22: List[str] = []
    colors_24: List[str] = []
    op_centers: List[float] = []
    separator_xs: List[float] = []
    drawn_ops: List[str] = []

    cursor = 1.0
    for op in operators:
        v22 = (
            df_2022.loc[
                (df_2022["operator"] == op) & (df_2022["rat"] == rat),
                metric["col_2022"],
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        v24 = (
            df_2024.loc[
                (df_2024["operator"] == op) & (df_2024["rat"] == rat),
                metric["col_2024"],
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        if v22.size == 0 and v24.size == 0:
            continue

        color = OP_COLORS.get(op, "lightgray")
        pos22 = cursor
        pos24 = cursor + pair_gap
        if v22.size:
            positions_22.append(pos22)
            data_22.append(v22)
            colors_22.append(color)
        if v24.size:
            positions_24.append(pos24)
            data_24.append(v24)
            colors_24.append(color)

        op_centers.append((pos22 + pos24) / 2.0)
        drawn_ops.append(op)
        cursor = pos24 + group_gap
        separator_xs.append(cursor - group_gap / 2.0)

    if not drawn_ops:
        print(f"[WARN] no data for all-ops box {rat} {metric_name}")
        return
    separator_xs = separator_xs[:-1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    if data_22:
        bp22 = ax.boxplot(
            data_22,
            positions=positions_22,
            widths=width,
            showfliers=False,
            patch_artist=True,
            whis=(5, 95),
        )
        for box, c in zip(bp22["boxes"], colors_22):
            box.set_facecolor(c)
            box.set_edgecolor("black")
        for median in bp22["medians"]:
            median.set_color("black")
    if data_24:
        bp24 = ax.boxplot(
            data_24,
            positions=positions_24,
            widths=width,
            showfliers=False,
            patch_artist=True,
            whis=(5, 95),
        )
        for box, c in zip(bp24["boxes"], colors_24):
            box.set_facecolor(c)
            box.set_edgecolor("black")
            box.set_hatch("////")
        for median in bp24["medians"]:
            median.set_color("black")

    for x in separator_xs:
        ax.axvline(x=x, linestyle="--", linewidth=0.8, color="0.35", alpha=0.9)

    ax.set_xticks(op_centers)
    ax.set_xticklabels([OP_LABELS.get(op, op.upper()) for op in drawn_ops])
    ax.set_xlim(0.0, cursor - group_gap + 1.0)
    ax.set_ylim(*metric["xlim"])
    ax.set_ylabel(metric["xlabel"])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    if metric_name == LEGEND_FIGURE:
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="lightgray", edgecolor="black", label="2022"),
                mpatches.Patch(
                    facecolor="lightgray", edgecolor="black", hatch="////", label="2024"
                ),
            ],
            loc="best",
        )
    _save(out_path)
    plt.close(fig)


def plot_op_rat_metric(
    op: str,
    rat: str,
    metric_name: str,
    df_2022: pd.DataFrame,
    df_2024: pd.DataFrame,
    out_path: Path,
) -> None:
    metric = METRICS[metric_name]
    color = OP_COLORS.get(op, "black")

    vals_2022 = (
        df_2022.loc[
            (df_2022["operator"] == op) & (df_2022["rat"] == rat), metric["col_2022"]
        ]
        .dropna()
        .to_numpy(dtype=float)
    )
    vals_2024 = (
        df_2024.loc[
            (df_2024["operator"] == op) & (df_2024["rat"] == rat), metric["col_2024"]
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    xs22, ys22 = build_cdf(vals_2022)
    xs24, ys24 = build_cdf(vals_2024)
    if xs22.size == 0 and xs24.size == 0:
        print(f"[WARN] no data for {op} {rat} {metric_name}")
        return

    plt.figure(figsize=FIG_SIZE)
    if xs22.size:
        plt.plot(xs22, ys22, color=color, label="2022", **YEAR_STYLES["2022"])
    if xs24.size:
        plt.plot(xs24, ys24, color=color, label="2024", **YEAR_STYLES["2024"])

    plt.xlabel(metric["xlabel"])
    plt.ylabel("CDF")
    plt.xlim(*metric["xlim"])
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    if op == LEGEND_OP:
        plt.legend()
    _save(out_path)
    plt.close()


def compute_stats(
    df_2022: pd.DataFrame,
    df_2024: pd.DataFrame,
    operators: List[str],
) -> pd.DataFrame:
    """Per (year, operator, rat, metric) summary stats matching the box/CDF plots."""
    rows = []
    for year, df in (("2022", df_2022), ("2024", df_2024)):
        for op in operators:
            for rat in RAT_ORDER:
                for metric_name, metric in METRICS.items():
                    col = metric["col_2022"] if year == "2022" else metric["col_2024"]
                    vals = (
                        df.loc[(df["operator"] == op) & (df["rat"] == rat), col]
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if vals.size == 0:
                        rows.append(
                            {
                                "year": year,
                                "operator": op,
                                "rat": rat,
                                "metric": metric_name,
                                "count": 0,
                            }
                        )
                        continue
                    p5, q1, median, q3, p95 = np.nanpercentile(
                        vals, [5, 25, 50, 75, 95]
                    )
                    rows.append(
                        {
                            "year": year,
                            "operator": op,
                            "rat": rat,
                            "metric": metric_name,
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
    ap = argparse.ArgumentParser(
        description="Overlay 2022 vs 2024 PCell metric CDFs per (operator, RAT)"
    )
    ap.add_argument(
        "--csv-2022",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2022/per_cell_all.csv",
    )
    ap.add_argument(
        "--cache-2024",
        type=Path,
        default=REPO_ROOT / "data/intermediate_data/2024/overall_interference",
        help="Directory holding the 2024 split cache CSVs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plots",
    )
    ap.add_argument(
        "--stats-csv", type=Path,
        default=REPO_ROOT / "data/intermediate_data/2022/fig_10_stats.csv",
    )
    ap.add_argument("--operators", nargs="+", default=OPS)
    args = ap.parse_args()

    operators = [str(o).lower() for o in args.operators if str(o).lower() in OPS]
    if not operators:
        raise SystemExit("No supported operators requested")

    print(f"[INFO] loading 2022: {args.csv_2022}")
    df22 = load_2022(args.csv_2022)
    df22 = df22[df22["operator"].isin(operators) & df22["rat"].isin(RAT_ORDER)]
    print(f"[INFO] 2022 rows={len(df22):,}")

    print(f"[INFO] loading 2024 cache: {args.cache_2024}")
    df24 = load_2024(args.cache_2024, operators)
    df24 = df24[df24["operator"].isin(operators) & df24["rat"].isin(RAT_ORDER)]
    print(f"[INFO] 2024 rows={len(df24):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats_df = compute_stats(df22, df24, operators)
    stats_csv = args.stats_csv
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_csv, index=False)
    print(f"[OK] Wrote {stats_csv}")

    for letter, rat, metric_name in (
        ("a", "LTE", "rsrp"), ("b", "LTE", "interference"), ("c", "LTE", "sinr"),
        ("d", "NR", "rsrp"), ("e", "NR", "interference"), ("f", "NR", "sinr"),
    ):
        plot_all_ops_rat_metric_box(
            rat, metric_name, df22, df24, operators,
            args.out_dir / f"fig_10{letter}.pdf",
        )


if __name__ == "__main__":
    main()
