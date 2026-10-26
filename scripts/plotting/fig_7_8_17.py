#!/usr/bin/env python3
"""Channel-level analysis"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_FONT_SIZE = 20
AXIS_FONT_SIZE = 18
LEGEND_FONT_SIZE = 15

def channel_figure_size(group_count: int, *, paired_metrics: bool) -> tuple[float, float]:
    width_per_group = 1.1 if paired_metrics else 1.0
    return (max(5.0, width_per_group * group_count), 5.0 if group_count >= 10 else 3.0)

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

MODE = "band"  # "freq", "area", "band", or "band_type"
SOURCE = "ue"  # "cell" or "ue"
FREQUENCY_LABEL = "n41-f8-T"
BAND_TYPE_GROUP_LABEL = FREQUENCY_LABEL  # Legacy CLI/default name.
OPERATOR = "tmobile"
AREA_TYPE = "rural"
BAND = "Band 66"
BAND_TYPE = "mid"

ROOT = Path(__file__).resolve().parents[2]
CELL_INPUT_CSV = ROOT / "data/intermediate_data/2024/all_cells_aggregated_report.csv"
UE_INPUT_CSV = ROOT / "data/intermediate_data/2024/complete_ue_level_metrics_report.csv"
UE_SUMMARY_CSV = ROOT / "data/intermediate_data/2024/complete_ue_level_summary.csv"
LTE_NEIGHBOR_SUMMARY_CSV = ROOT / "data/intermediate_data/2024/lte_channel_neighbor_summary.csv"
OUT_DIR = ROOT / "plots"

AREA_ORDER = ["rural", "suburban", "urban"]
MIN_CELLS_PER_GROUP = 20
UE_REQUIRED_COLUMNS = {
    "operator",
    "rat",
    "band",
    "band_type",
    "channel_key",
    "frequency_label",
    "area_type",
    "pcell_key",
    "rsrp",
    "interference",
    "sinr",
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def strip_band_prefix(label: Any) -> str:
    s = str(label)
    return s.split("-", 1)[1] if "-" in s else s


def require_columns(df: pd.DataFrame, cols: set[str], path: Path) -> None:
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")


def default_input_csv(source: str) -> Path:
    return UE_INPUT_CSV if source == "ue" else CELL_INPUT_CSV


def clean_join_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        num = float(text)
    except ValueError:
        return text
    if np.isfinite(num) and num.is_integer():
        return str(int(num))
    return text


def load_lte_avg_neighbor_lookup(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df["operator"] = df["operator"].map(clean_join_value)
    df["channel_key"] = df["channel_key"].map(clean_join_value)
    df["avg_neighbor_count"] = pd.to_numeric(df["avg_neighbor_count"], errors="coerce")
    df = df.dropna(subset=["avg_neighbor_count"])
    return {
        (op, ch): float(val)
        for op, ch, val in zip(
            df["operator"], df["channel_key"], df["avg_neighbor_count"]
        )
    }


def override_lte_avg_interferers(df: pd.DataFrame) -> pd.DataFrame:
    """For LTE rows, replace avg_interferers with values from the per-channel
    LTE neighbor summary (source: raw Qualcomm LTE intrafreq logs). NR rows
    are left untouched and continue to use the legacy timing-log neighbor count."""
    if df.empty or "rat" not in df.columns:
        return df
    lookup = load_lte_avg_neighbor_lookup(LTE_NEIGHBOR_SUMMARY_CSV)
    if not lookup:
        return df
    rat_upper = df["rat"].astype(str).str.upper().str.strip()
    is_lte = rat_upper.isin(["4G", "LTE"])
    if not is_lte.any():
        return df
    if "avg_interferers" not in df.columns:
        df["avg_interferers"] = np.nan
    keys = list(
        zip(
            df.loc[is_lte, "operator"].map(clean_join_value),
            df.loc[is_lte, "channel_key"].map(clean_join_value),
        )
    )
    df.loc[is_lte, "avg_interferers"] = [lookup.get(k, np.nan) for k in keys]
    return df


def normalize_join_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for key in keys:
        if key in out.columns:
            out[key] = out[key].map(clean_join_value)
    return out


def normalize_ue_dataframe(
    df: pd.DataFrame, input_csv: Path, ue_summary_csv: Path
) -> pd.DataFrame:
    require_columns(
        df,
        UE_REQUIRED_COLUMNS,
        input_csv,
    )
    summary = pd.read_csv(ue_summary_csv)
    require_columns(
        summary,
        {
            "operator",
            "rat",
            "band_type",
            "band",
            "channel_key",
            "frequency_label",
            "area_type",
            "ue_avg_interferers",
        },
        ue_summary_csv,
    )

    join_keys = [
        "operator",
        "rat",
        "band_type",
        "band",
        "channel_key",
        "frequency_label",
        "area_type",
    ]
    df = normalize_join_keys(df, join_keys)
    summary = normalize_join_keys(summary, join_keys)
    summary = summary[join_keys + ["ue_avg_interferers"]].drop_duplicates(join_keys)
    out = df.merge(summary, on=join_keys, how="left", validate="many_to_one")
    out = out.rename(
        columns={
            "rsrp": "avg_rsrp",
            "interference": "avg_interference",
            "sinr": "avg_sinr",
            "ue_avg_interferers": "avg_interferers",
        }
    )
    out = override_lte_avg_interferers(out)
    return out


def load_ue_filtered_dataframe(
    input_csv: Path,
    ue_summary_csv: Path,
    operator: Optional[str] = None,
    band: Optional[str] = None,
    band_type: Optional[str] = None,
    frequency_label: Optional[str] = None,
    require_area_type: bool = False,
    require_frequency_label: bool = False,
) -> pd.DataFrame:
    parts = []
    usecols = sorted(UE_REQUIRED_COLUMNS)
    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=400_000):
        mask = pd.Series(True, index=chunk.index)
        if operator is not None:
            mask &= chunk["operator"].astype(str).str.lower() == str(operator).lower()
        if band is not None:
            mask &= chunk["band"].astype(str) == str(band)
        if band_type is not None:
            mask &= chunk["band_type"].astype(str) == str(band_type)
        if frequency_label is not None:
            mask &= chunk["frequency_label"].astype(str) == str(frequency_label)
        if require_area_type:
            mask &= chunk["area_type"].notna()
        if require_frequency_label:
            mask &= chunk["frequency_label"].notna()
        if mask.any():
            parts.append(chunk.loc[mask].copy())

    if parts:
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.DataFrame(columns=usecols)
    return normalize_ue_dataframe(df, input_csv, ue_summary_csv)


def load_input_dataframe(
    input_csv: Path, source: str, ue_summary_csv: Path
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if source == "cell":
        require_columns(
            df,
            {
                "operator",
                "band",
                "frequency_label",
                "area_type",
                "pcell_key",
                "avg_interferers",
                "avg_rsrp",
                "avg_interference",
                "avg_sinr",
            },
            input_csv,
        )
        df = override_lte_avg_interferers(df)
        return df
    return normalize_ue_dataframe(df, input_csv, ue_summary_csv)


def area_sort_key(area_type: Any) -> tuple[int, str]:
    s = str(area_type)
    if s in AREA_ORDER:
        return (AREA_ORDER.index(s), s)
    return (len(AREA_ORDER), s)


def collect_box_data(
    df: pd.DataFrame, group_col: str, group_order: list[str], metric_col: str
) -> tuple[list[np.ndarray], list[str]]:
    data = []
    labels = []
    for label in group_order:
        vals = pd.to_numeric(
            df.loc[df[group_col].astype(str) == label, metric_col],
            errors="coerce",
        ).to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(label)
    return data, labels


def filter_groups_by_min_cells(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    min_cells: int = MIN_CELLS_PER_GROUP,
) -> tuple[pd.DataFrame, list[str]]:
    counts = df.groupby(group_col)["pcell_key"].nunique()
    keep = set(counts[counts >= int(min_cells)].index.astype(str))
    filtered_order = [label for label in group_order if str(label) in keep]
    filtered_df = df[df[group_col].astype(str).isin(filtered_order)].copy()
    dropped = [label for label in group_order if str(label) not in keep]
    if dropped:
        print(
            f"[INFO] Dropped groups with < {min_cells} cells: "
            f"{', '.join(map(str, dropped))}"
        )
    return filtered_df, filtered_order


def plot_rsrp_interference(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title_prefix: str,
    x_axis_name: str,
    out_png: Path,
    x_rotation: int = 0,
    show_avg_interferers: bool = True,
) -> None:
    rsrp_data, rsrp_labels = collect_box_data(df, group_col, group_order, "avg_rsrp")
    int_data, int_labels = collect_box_data(
        df, group_col, group_order, "avg_interference"
    )
    common_labels = [
        label for label in group_order if label in rsrp_labels and label in int_labels
    ]
    rsrp_data, labels = collect_box_data(df, group_col, common_labels, "avg_rsrp")
    int_data, _ = collect_box_data(df, group_col, common_labels, "avg_interference")
    if not rsrp_data or not int_data:
        print("[WARN] Not enough RSRP/interference data to plot.")
        return

    x = np.arange(1, len(labels) + 1)
    offset = 0.18
    fig, ax = plt.subplots(figsize=channel_figure_size(len(labels), paired_metrics=True))

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
        box.set_hatch("////")
        box.set_facecolor("lightblue")
        box.set_edgecolor("black")
    for box in bp2["boxes"]:
        box.set_hatch("\\\\\\\\")
        box.set_facecolor("salmon")
        box.set_edgecolor("black")
    for line in bp1["medians"] + bp2["medians"]:
        line.set_color("black")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [strip_band_prefix(l) for l in labels],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("Power (dBm)")
    ax.grid(True, axis="y", alpha=0.3)

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
        loc="upper left",
    )

    ax.set_ylim(-130.0, -60.0 if len(labels) >= 10 else -40.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_png}")


def plot_sinr(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    title_prefix: str,
    x_axis_name: str,
    out_png: Path,
    x_rotation: int = 0,
    show_avg_interferers: bool = True,
) -> None:
    data, labels = collect_box_data(df, group_col, group_order, "avg_sinr")
    if not data:
        print("[WARN] Not enough SINR data to plot.")
        return

    fig, ax = plt.subplots(figsize=channel_figure_size(len(labels), paired_metrics=False))
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#dbeafe")
        box.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(
        [strip_band_prefix(l) for l in labels],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("SINR (dB)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-10.0, 40.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_png}")


def natural_label_key(label: Any) -> list[Any]:
    parts = re.split(r"(\d+)", str(label))
    return [int(p) if p.isdigit() else p for p in parts]


def write_counts_table_tex(
    df: pd.DataFrame,
    group_col: str,
    out_tex: Path,
    group_order: Optional[list[str]] = None,
) -> None:
    if group_order is None:
        group_order = sorted(
            df[group_col].dropna().astype(str).unique(),
            key=natural_label_key,
        )
    freq_cells: list[str] = []
    count_cells: list[str] = []
    interferer_cells: list[str] = []
    for label in group_order:
        group_rows = df.loc[df[group_col].astype(str) == label]
        n_cells = group_rows["pcell_key"].nunique()
        avg_interferers = pd.to_numeric(
            group_rows["avg_interferers"], errors="coerce"
        ).mean()
        freq_cells.append(strip_band_prefix(label))
        count_cells.append(str(int(n_cells)))
        interferer_cells.append(
            "NA" if pd.isna(avg_interferers) else f"{avg_interferers:.2f}"
        )

    col_spec = "l|" + "c" * len(group_order)
    lines = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\hline",
        "Frequency & " + " & ".join(freq_cells) + " \\\\",
        "\\hline",
        "Cells & " + " & ".join(count_cells) + " \\\\",
        "Interfering cells & " + " & ".join(interferer_cells) + " \\\\",
        "\\hline",
        "\\end{tabular}%",
        "}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[OK] Wrote {out_tex}")


def load_filtered_cells(
    input_csv: Path, source: str, ue_summary_csv: Path, label: str
) -> pd.DataFrame:
    if source == "ue":
        df = load_ue_filtered_dataframe(
            input_csv,
            ue_summary_csv,
            frequency_label=label,
            require_area_type=True,
        )
    else:
        df = load_input_dataframe(input_csv, source, ue_summary_csv)
    require_columns(
        df,
        {
            "operator",
            "band",
            "frequency_label",
            "area_type",
            "pcell_key",
            "avg_interferers",
            "avg_rsrp",
            "avg_interference",
            "avg_sinr",
        },
        input_csv,
    )
    sub = df[df["frequency_label"].astype(str) == str(label)].copy()
    sub = sub.dropna(subset=["area_type"])
    for col in ["avg_interferers", "avg_rsrp", "avg_interference", "avg_sinr"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub


def load_area_mode_cells(
    input_csv: Path, source: str, ue_summary_csv: Path, operator: str, band: str
) -> pd.DataFrame:
    if source == "ue":
        df = load_ue_filtered_dataframe(
            input_csv,
            ue_summary_csv,
            operator=operator,
            band=band,
            require_area_type=True,
        )
    else:
        df = load_input_dataframe(input_csv, source, ue_summary_csv)
    require_columns(
        df,
        {
            "operator",
            "band",
            "area_type",
            "pcell_key",
            "avg_interferers",
            "avg_rsrp",
            "avg_interference",
            "avg_sinr",
        },
        input_csv,
    )
    sub = df[
        (df["operator"].astype(str).str.lower() == str(operator).lower())
        & (df["band"].astype(str) == str(band))
        & df["area_type"].notna()
    ].copy()
    for col in ["avg_interferers", "avg_rsrp", "avg_interference", "avg_sinr"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub


def load_band_mode_cells(
    input_csv: Path, source: str, ue_summary_csv: Path, operator: str, band: str
) -> pd.DataFrame:
    if source == "ue":
        df = load_ue_filtered_dataframe(
            input_csv,
            ue_summary_csv,
            operator=operator,
            band=band,
            require_frequency_label=True,
        )
    else:
        df = load_input_dataframe(input_csv, source, ue_summary_csv)
    require_columns(
        df,
        {
            "operator",
            "band",
            "frequency_label",
            "pcell_key",
            "avg_interferers",
            "avg_rsrp",
            "avg_interference",
            "avg_sinr",
        },
        input_csv,
    )
    sub = df[
        (df["operator"].astype(str).str.lower() == str(operator).lower())
        & (df["band"].astype(str) == str(band))
        & df["frequency_label"].notna()
    ].copy()
    for col in ["avg_interferers", "avg_rsrp", "avg_interference", "avg_sinr"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub


def load_band_type_mode_cells(
    input_csv: Path,
    source: str,
    ue_summary_csv: Path,
    operator: str,
    band_type: str,
) -> pd.DataFrame:
    if source == "ue":
        df = load_ue_filtered_dataframe(
            input_csv,
            ue_summary_csv,
            operator=operator,
            band_type=band_type,
            require_frequency_label=True,
        )
    else:
        df = load_input_dataframe(input_csv, source, ue_summary_csv)
    require_columns(
        df,
        {
            "operator",
            "band_type",
            "frequency_label",
            "pcell_key",
            "avg_interferers",
            "avg_rsrp",
            "avg_interference",
            "avg_sinr",
        },
        input_csv,
    )
    sub = df[
        (df["operator"].astype(str).str.lower() == str(operator).lower())
        & (df["band_type"].astype(str) == str(band_type))
        & df["frequency_label"].notna()
    ].copy()
    for col in ["avg_interferers", "avg_rsrp", "avg_interference", "avg_sinr"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub


def describe_label(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return str(label)
    row = df.iloc[0]
    return (
        f"{label} ({str(row['operator']).upper()}, "
        f"{row['band']}, {row['frequency_label']})"
    )


def group_order_by_median_rsrp(df: pd.DataFrame, group_col: str) -> list[str]:
    groups = df.groupby(group_col).agg(median_rsrp=("avg_rsrp", "median")).reset_index()
    groups = groups.sort_values(
        ["median_rsrp", group_col],
        ascending=[True, True],
        na_position="last",
    )
    return groups[group_col].astype(str).tolist()


def group_order_by_avg_interferers(df: pd.DataFrame, group_col: str) -> list[str]:
    groups = (
        df.groupby(group_col)
        .agg(avg_interferers=("avg_interferers", "mean"))
        .reset_index()
    )
    groups = groups.sort_values(
        ["avg_interferers", group_col],
        ascending=[True, True],
        na_position="last",
    )
    return groups[group_col].astype(str).tolist()


def rat_dir_for_band(band: str) -> str:
    """NR bands are labelled like 'n77'; everything else is treated as LTE."""
    return "NR" if str(band).strip().lower().startswith("n") else "LTE"


def run_band_ue(
    operator: str,
    band: str,
    input_csv: Path,
    ue_summary_csv: Path,
    out_dir: Path,
) -> None:
    cells = load_band_mode_cells(input_csv, "ue", ue_summary_csv, operator, band)
    if cells.empty:
        print(f"[WARN] No rows for operator={operator!r}, band={band!r}; skipping.")
        return

    group_col = "frequency_label"
    group_order = group_order_by_avg_interferers(cells, group_col)
    cells, group_order = filter_groups_by_min_cells(
        cells, group_col, group_order, min_cells=MIN_CELLS_PER_GROUP
    )
    if cells.empty or not group_order:
        print(
            f"[WARN] No {group_col} groups with >= {MIN_CELLS_PER_GROUP} cells "
            f"for operator={operator!r}, band={band!r}; skipping."
        )
        return

    panel = {
        ("att", "n77"): (7, "a", "b"),
        ("tmobile", "n41"): (7, "c", "d"),
        ("verizon", "n77"): (7, "e", "f"),
        ("verizon", "Band 66"): (8, "a", "b"),
        ("att", "Band 2"): (17, "a", "b"),
        ("tmobile", "Band 66"): (17, "c", "d"),
    }[(operator, band)]
    figure, rsrp_letter, sinr_letter = panel
    title_prefix = f"{operator.upper()} {band}"
    x_axis_name = "Frequency Label"
    x_rotation = 60

    plot_rsrp_interference(
        cells, group_col, group_order, title_prefix, x_axis_name,
        out_dir / f"fig_{figure}{rsrp_letter}.pdf",
        x_rotation=x_rotation, show_avg_interferers=True,
    )
    plot_sinr(
        cells, group_col, group_order, title_prefix, x_axis_name,
        out_dir / f"fig_{figure}{sinr_letter}.pdf",
        x_rotation=x_rotation, show_avg_interferers=True,
    )
    print(
        f"[INFO] {operator} {band} -> {rat_dir_for_band(band)}: "
        f"rows={len(cells)}, cells={cells['pcell_key'].nunique()}"
    )


def run_batch(input_csv: Path, ue_summary_csv: Path, out_dir: Path) -> None:
    for operator, band in (
        ("att", "n77"), ("tmobile", "n41"), ("verizon", "n77"),
        ("verizon", "Band 66"), ("att", "Band 2"), ("tmobile", "Band 66"),
    ):
        run_band_ue(operator, band, input_csv, ue_summary_csv, out_dir)


def main() -> None:
    global LTE_NEIGHBOR_SUMMARY_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=UE_INPUT_CSV)
    parser.add_argument("--ue-summary-csv", type=Path, default=UE_SUMMARY_CSV)
    parser.add_argument("--lte-neighbor-summary-csv", type=Path, default=LTE_NEIGHBOR_SUMMARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    LTE_NEIGHBOR_SUMMARY_CSV = args.lte_neighbor_summary_csv
    run_batch(args.input_csv, args.ue_summary_csv, args.out_dir)


if __name__ == "__main__":
    main()
