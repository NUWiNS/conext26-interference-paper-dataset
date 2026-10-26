#!/usr/bin/env python3
"""User performance scatter plots"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Global font and figure-size settings for all figures.
LABEL_FONT_SIZE = 27
AXIS_FONT_SIZE = 25
LEGEND_FONT_SIZE = 25
FIG_SIZE = (5, 4)
FIG_SIZE_YLABEL = (5.4, 4)

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

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "data/raw_data/2024/app_data_all"
OUT_DIR = ROOT / "plots"
REPORT_DIR = ROOT / "data/intermediate_data/2024/mcs_tput_reports"
SCATTER_DIR = OUT_DIR

OPS = ["att", "tmobile", "verizon"]
OP_DISPLAY = {"att": "AT&T", "tmobile": "T-Mobile", "verizon": "Verizon"}
BAND_TYPE_ORDER = ["low", "mid", "mmwave"]
GRID_BAND_TYPES = ["low", "mid"]
GRID_PRB_SINR_MIN = 15.0
GRID_PRB_MCS_MIN = 10.0

PRB_BOX_CONFIGS = [
    ("mid", 20.0, 200.0),
    ("low", 10.0, 40.0),
]

PRB_TPUT_SCATTER_CONFIGS_1 = [
    ("mid", 15.0, 10.0),
    ("low", 15.0, 10.0),
    ("mmwave", 15.0, 10.0),
]

UTC_COL = "utc_ts"

LTE_PCI_COL = "LTE KPI PCell Serving PCI"
LTE_BAND_COL = "LTE KPI PCell Serving Band"
LTE_EARFCN_COL = "LTE KPI PCell Serving EARFCN(DL)"
LTE_RSRP_COL = "LTE KPI PCell Serving RSRP [dBm]"
LTE_SINR_COL = "LTE KPI PCell SINR [dB]"
LTE_MCS0_COL = "LTE KPI PCell DL MCS0"
LTE_MCS1_COL = "LTE KPI PCell DL MCS1"
LTE_TPUT_COL = "LTE KPI PCell MAC DL Throughput [Mbps]"
LTE_PRB_COL = "LTE KPI PCell PDSCH PRB Number(Including 0)"

NR_PCI_COL = "5G KPI PCell RF Serving PCI"
NR_ARFCN_COL = "5G KPI PCell RF NR-ARFCN"
NR_FREQ_COL = "5G KPI PCell RF Frequency [MHz]"
NR_RSRP_COL = "5G KPI PCell RF Serving SS-RSRP [dBm]"
NR_SINR_COL = "5G KPI PCell RF Serving SS-SINR [dB]"
NR_MCS_COL = "5G KPI PCell Layer1 DL MCS (Avg)"
NR_TPUT_COL = "5G KPI PCell Layer2 MAC DL Throughput [Mbps]"
NR_PRB_COL = "5G KPI PCell Layer1 DL RB Num (Including 0)"

REQUIRED_COLS = [
    UTC_COL,
    LTE_PCI_COL,
    LTE_BAND_COL,
    LTE_EARFCN_COL,
    LTE_RSRP_COL,
    LTE_SINR_COL,
    LTE_MCS0_COL,
    LTE_MCS1_COL,
    LTE_TPUT_COL,
    LTE_PRB_COL,
    NR_PCI_COL,
    NR_ARFCN_COL,
    NR_FREQ_COL,
    NR_RSRP_COL,
    NR_SINR_COL,
    NR_MCS_COL,
    NR_TPUT_COL,
    NR_PRB_COL,
]

REPORT_COLUMNS = [
    "operator",
    "date",
    "time_dir",
    "source_file",
    "source_row_index",
    "utc_ts",
    "rat",
    "band",
    "band_type",
    "pci",
    "earfcn",
    "arfcn",
    "frequency_mhz",
    "rsrp_dbm",
    "sinr_db",
    "interference_db",
    "mcs",
    "throughput_mbps",
    "prb_num",
    "lte_mcs0",
    "lte_mcs1",
]

LTE_BAND_TABLE = {
    1: (2110.0, 0, 0, 599),
    2: (1930.0, 600, 600, 1199),
    3: (1805.0, 1200, 1200, 1949),
    4: (2110.0, 1950, 1950, 2399),
    5: (869.0, 2400, 2400, 2649),
    7: (2620.0, 2750, 2750, 3449),
    8: (925.0, 3450, 3450, 3799),
    12: (729.0, 5010, 5010, 5179),
    13: (746.0, 5180, 5180, 5279),
    14: (758.0, 5280, 5280, 5379),
    17: (734.0, 5730, 5730, 5849),
    18: (860.0, 5850, 5850, 5999),
    19: (875.0, 6000, 6000, 6149),
    20: (791.0, 6150, 6150, 6449),
    25: (1930.0, 8040, 8040, 8689),
    26: (859.0, 8690, 8690, 9039),
    28: (758.0, 9210, 9210, 9659),
    29: (717.0, 9660, 9660, 9769),
    30: (2350.0, 9770, 9770, 9869),
    32: (1452.0, 9920, 9920, 10359),
    34: (2010.0, 36200, 36200, 36349),
    38: (2570.0, 37750, 37750, 38249),
    39: (1880.0, 38250, 38250, 38649),
    40: (2300.0, 38650, 38650, 39649),
    41: (2496.0, 39650, 39650, 41589),
    66: (2110.0, 66436, 66436, 67335),
    71: (617.0, 68586, 68586, 68935),
}


def clean_value(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def to_float(value) -> Optional[float]:
    s = clean_value(value)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def format_number(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.6g}"


def normalize_lte_band(band) -> Optional[int]:
    s = clean_value(band)
    if s is None:
        return None
    m = re.search(r"(?:band\s*|b)?(\d+)", s, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def lte_dl_freq_mhz(band, earfcn) -> Optional[float]:
    band_num = normalize_lte_band(band)
    n = to_float(earfcn)
    if band_num is None or n is None or band_num not in LTE_BAND_TABLE:
        return None
    f_low, n_offs, n_min, n_max = LTE_BAND_TABLE[band_num]
    if not (n_min <= n <= n_max):
        return None
    return f_low + 0.1 * (n - n_offs)


def band_type_from_freq(freq_mhz: Optional[float]) -> Optional[str]:
    if freq_mhz is None or math.isnan(freq_mhz):
        return None
    if freq_mhz < 1000:
        return "low"
    if freq_mhz < 6000:
        return "mid"
    if freq_mhz >= 24000:
        return "mmwave"
    return None


def lte_mcs(mcs0, mcs1) -> Optional[float]:
    m0 = to_float(mcs0)
    m1 = to_float(mcs1)
    if m0 is not None and m1 is not None:
        return (m0 + m1) / 2.0
    if m0 is not None:
        return m0
    return None


def iter_downlink_files(input_root: Path, operators: Iterable[str]) -> Iterable[Path]:
    for op in operators:
        op_dir = input_root / op
        if not op_dir.exists():
            print(f"[WARN] Missing operator directory: {op_dir}")
            continue
        yield from sorted(op_dir.glob("*/" "*/tcp_downlink*xcal.csv"))


def parse_path_parts(path: Path, input_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(input_root)
    op = rel.parts[0] if len(rel.parts) > 0 else ""
    date = rel.parts[1] if len(rel.parts) > 1 else ""
    time_dir = rel.parts[2] if len(rel.parts) > 2 else ""
    return op, date, time_dir


def validate_header(header: list[str], path: Path) -> None:
    missing = [col for col in REQUIRED_COLS if col not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


class SplitCsvWriter:
    def __init__(self, out_dir: Path, stem: str, fieldnames: list[str], max_bytes: int):
        self.out_dir = out_dir
        self.stem = stem
        self.fieldnames = fieldnames
        self.max_bytes = max_bytes
        self.part = 0
        self.current_bytes = 0
        self.current_rows = 0
        self.file = None
        self.writer = None
        self.paths: list[Path] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self) -> None:
        self.close()
        self.part += 1
        path = self.out_dir / f"{self.stem}_part{self.part:02d}.csv"
        self.file = path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.paths.append(path)
        header_buf = io.StringIO()
        csv.DictWriter(header_buf, fieldnames=self.fieldnames).writeheader()
        self.current_bytes = len(header_buf.getvalue().encode("utf-8"))
        self.current_rows = 0

    def _row_bytes(self, row: dict[str, object]) -> int:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.fieldnames)
        writer.writerow(row)
        return len(buf.getvalue().encode("utf-8"))

    def write_row(self, row: dict[str, object]) -> None:
        if self.writer is None:
            self._open_next()
        row_bytes = self._row_bytes(row)
        if self.current_rows > 0 and self.current_bytes + row_bytes > self.max_bytes:
            self._open_next()
        assert self.writer is not None
        self.writer.writerow(row)
        self.current_bytes += row_bytes
        self.current_rows += 1

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None


def make_report_row(
    op: str,
    date: str,
    time_dir: str,
    path: Path,
    row_index: int,
    raw_row: dict[str, str],
    rat: str,
    band: Optional[str],
    band_type: Optional[str],
    pci: Optional[str],
    earfcn: Optional[str],
    arfcn: Optional[str],
    freq_mhz: Optional[float],
    rsrp: float,
    sinr: float,
    mcs: Optional[float],
    tput: Optional[float],
    prb: Optional[float],
) -> dict[str, object]:
    return {
        "operator": op,
        "date": date,
        "time_dir": time_dir,
        "source_file": path.name,
        "source_row_index": row_index,
        "utc_ts": clean_value(raw_row.get(UTC_COL)) or "",
        "rat": rat,
        "band": band or "",
        "band_type": band_type or "",
        "pci": pci or "",
        "earfcn": earfcn or "",
        "arfcn": arfcn or "",
        "frequency_mhz": format_number(freq_mhz),
        "rsrp_dbm": format_number(rsrp),
        "sinr_db": format_number(sinr),
        "interference_db": format_number(rsrp - sinr),
        "mcs": format_number(mcs),
        "throughput_mbps": format_number(tput),
        "prb_num": format_number(prb),
        "lte_mcs0": clean_value(raw_row.get(LTE_MCS0_COL)) or "",
        "lte_mcs1": clean_value(raw_row.get(LTE_MCS1_COL)) or "",
    }


def process_raw_data(
    input_root: Path,
    report_dir: Path,
    operators: list[str],
    max_report_mb: float,
) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    for old in report_dir.glob("mcs_tput_report_part*.csv"):
        old.unlink()

    writer = SplitCsvWriter(
        report_dir,
        "mcs_tput_report",
        REPORT_COLUMNS,
        max_bytes=int(max_report_mb * 1000 * 1000),
    )
    fiveg_rows: list[dict[str, object]] = []
    counts = {op: {"LTE": 0, "5G": 0} for op in operators}

    files = list(iter_downlink_files(input_root, operators))
    if not files:
        raise SystemExit(f"No tcp_downlink*xcal.csv files found under {input_root}")

    try:
        for file_idx, path in enumerate(files, start=1):
            op, date, time_dir = parse_path_parts(path, input_root)
            last_lte_band = None
            last_lte_earfcn = None
            last_nr_freq = None

            with path.open(newline="", errors="replace") as f:
                reader = csv.DictReader(f)
                validate_header(reader.fieldnames or [], path)

                for row_index, row in enumerate(reader):
                    current_lte_band = clean_value(row.get(LTE_BAND_COL))
                    current_lte_earfcn = clean_value(row.get(LTE_EARFCN_COL))
                    current_nr_freq = to_float(row.get(NR_FREQ_COL))
                    if current_lte_band is not None:
                        last_lte_band = current_lte_band
                    if current_lte_earfcn is not None:
                        last_lte_earfcn = current_lte_earfcn
                    if current_nr_freq is not None:
                        last_nr_freq = current_nr_freq

                    lte_rsrp = to_float(row.get(LTE_RSRP_COL))
                    lte_sinr = to_float(row.get(LTE_SINR_COL))
                    if lte_rsrp is not None and lte_sinr is not None:
                        lte_freq = lte_dl_freq_mhz(last_lte_band, last_lte_earfcn)
                        lte_type = band_type_from_freq(lte_freq)
                        report_row = make_report_row(
                            op=op,
                            date=date,
                            time_dir=time_dir,
                            path=path,
                            row_index=row_index,
                            raw_row=row,
                            rat="LTE",
                            band=last_lte_band,
                            band_type=lte_type,
                            pci=clean_value(row.get(LTE_PCI_COL)),
                            earfcn=last_lte_earfcn,
                            arfcn=None,
                            freq_mhz=lte_freq,
                            rsrp=lte_rsrp,
                            sinr=lte_sinr,
                            mcs=lte_mcs(row.get(LTE_MCS0_COL), row.get(LTE_MCS1_COL)),
                            tput=to_float(row.get(LTE_TPUT_COL)),
                            prb=to_float(row.get(LTE_PRB_COL)),
                        )
                        writer.write_row(report_row)
                        counts[op]["LTE"] += 1

                    nr_rsrp = to_float(row.get(NR_RSRP_COL))
                    nr_sinr = to_float(row.get(NR_SINR_COL))
                    if nr_rsrp is not None and nr_sinr is not None:
                        nr_type = band_type_from_freq(last_nr_freq)
                        report_row = make_report_row(
                            op=op,
                            date=date,
                            time_dir=time_dir,
                            path=path,
                            row_index=row_index,
                            raw_row=row,
                            rat="5G",
                            band=None,
                            band_type=nr_type,
                            pci=clean_value(row.get(NR_PCI_COL)),
                            earfcn=None,
                            arfcn=clean_value(row.get(NR_ARFCN_COL)),
                            freq_mhz=last_nr_freq,
                            rsrp=nr_rsrp,
                            sinr=nr_sinr,
                            mcs=to_float(row.get(NR_MCS_COL)),
                            tput=to_float(row.get(NR_TPUT_COL)),
                            prb=to_float(row.get(NR_PRB_COL)),
                        )
                        writer.write_row(report_row)
                        fiveg_rows.append(report_row)
                        counts[op]["5G"] += 1

            if file_idx % 250 == 0:
                print(f"[INFO] Parsed {file_idx}/{len(files)} files")
    finally:
        writer.close()

    print(f"[OK] Wrote {len(writer.paths)} report part(s) under {report_dir}")
    for op in operators:
        print(f"[INFO] {op}: LTE rows={counts[op]['LTE']}, 5G rows={counts[op]['5G']}")

    return pd.DataFrame(fiveg_rows, columns=REPORT_COLUMNS)


def load_5g_from_reports(report_dir: Path) -> pd.DataFrame:
    parts = sorted(report_dir.glob("mcs_tput_report_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"No report parts found under {report_dir}")
    frames = []
    usecols = [
        "operator",
        "rat",
        "band_type",
        "frequency_mhz",
        "rsrp_dbm",
        "sinr_db",
        "interference_db",
        "mcs",
        "throughput_mbps",
        "prb_num",
    ]
    for part in parts:
        df = pd.read_csv(part, usecols=usecols)
        df = df[df["rat"].astype(str) == "5G"].copy()
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def numeric_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "sinr_db",
        "interference_db",
        "mcs",
        "throughput_mbps",
        "prb_num",
        "frequency_mhz",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["operator"] = out["operator"].astype(str).str.lower()
    out["band_type"] = out["band_type"].astype(str).str.lower()
    out = out[out["band_type"].isin(BAND_TYPE_ORDER)].copy()
    return out


def stable_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=7)


def prb_tput_population(
    plot_df: pd.DataFrame, sinr_min_db: float, mcs_min: float
) -> pd.DataFrame:
    return plot_df[
        plot_df["sinr_db"].notna()
        & (plot_df["sinr_db"] > sinr_min_db)
        & plot_df["mcs"].notna()
        & (plot_df["mcs"] >= mcs_min)
        & plot_df["throughput_mbps"].notna()
        & plot_df["prb_num"].notna()
    ]


def prb_tput_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and tie-aware Spearman rho, separately for each scatter panel."""
    plot_df = numeric_plot_frame(df)
    rows = []
    for band_type, sinr_min, mcs_min in PRB_TPUT_SCATTER_CONFIGS_1:
        base = prb_tput_population(plot_df, sinr_min, mcs_min)
        for op in OPS:
            selected = base[
                (base["operator"] == op) & (base["band_type"] == band_type)
            ]
            finite = np.isfinite(selected[
                ["sinr_db", "mcs", "throughput_mbps", "prb_num"]
            ]).all(axis=1)
            sub = selected.loc[finite]
            x, y = sub["prb_num"], sub["throughput_mbps"]
            defined = len(sub) >= 2 and x.nunique() > 1 and y.nunique() > 1
            rows.append({
                "operator": op,
                "band_type": band_type,
                "sinr_min_exclusive_db": sinr_min,
                "mcs_min_inclusive": mcs_min,
                "n_eligible_rows": len(selected),
                "n_nonfinite_excluded": int((~finite).sum()),
                "n_samples": len(sub),
                "n_mcs_exactly_min": int((sub["mcs"] == mcs_min).sum()),
                "pearson_r": x.corr(y) if defined else np.nan,
                "spearman_rho": x.rank().corr(y.rank()) if defined else np.nan,
            })
    return pd.DataFrame(rows)


def write_prb_tput_correlations(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = prb_tput_correlations(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prb_tput_correlations.csv"
    summary.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")
    return summary


def plot_scatter(
    df: pd.DataFrame,
    op: str,
    band_type: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    max_points: int,
    show_yaxis: bool = True,
) -> None:
    sub = df[
        (df["operator"] == op)
        & (df["band_type"] == band_type)
        & df[x_col].notna()
        & df[y_col].notna()
    ].copy()
    if sub.empty:
        print(f"[WARN] No 5G {op} {band_type} rows for {y_col} vs {x_col}")
        return

    original_n = len(sub)
    sub = stable_sample(sub, max_points)

    fig, ax = plt.subplots(figsize=FIG_SIZE_YLABEL if show_yaxis else FIG_SIZE)
    ax.scatter(sub[x_col], sub[y_col], s=24, alpha=0.8, linewidths=0, rasterized=True)
    ax.set_xlabel(x_label)
    if show_yaxis:
        ax.set_ylabel(y_label)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(True, alpha=0.25)
    if x_col == "sinr_db":
        ax.set_xlim(-20, 40)
    elif x_col == "interference_db":
        ax.set_xlim(-140, -60)
    ax.margins(x=0, y=0)
    if y_col == "throughput_mbps":
        ax.set_ylim(0, 100 if band_type == "low" else 500)
    elif y_col == "mcs":
        ax.set_ylim(0, 30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def plot_5g_scatters(df: pd.DataFrame, out_dir: Path, max_points: int) -> None:
    plot_df = numeric_plot_frame(df)
    if plot_df.empty:
        print("[WARN] No 5G rows available for plotting.")
        return

    plot_df = plot_df[
        plot_df["mcs"].notna()
        & plot_df["throughput_mbps"].notna()
        & plot_df["prb_num"].notna()
    ]
    if plot_df.empty:
        print("[WARN] No 5G rows with mcs+throughput+prb all present.")
        return

    for op in ("att", "verizon"):
        for band_type in ("mmwave",):
            mcs_yaxis = band_type == "mmwave" or (op == "att" and band_type == "low")
            tput_yaxis = band_type == "mmwave" or op == "att"
            mcs_panel, tput_panel = (("a", "b") if op == "att" else ("d", "e"))
            plot_scatter(
                plot_df, op, band_type, "sinr_db", "mcs", "SINR (dB)", "DL MCS",
                out_dir / f"fig_16{mcs_panel}.pdf", max_points, show_yaxis=mcs_yaxis,
            )
            plot_scatter(
                plot_df, op, band_type, "sinr_db", "throughput_mbps", "SINR (dB)",
                "Tput (Mbps)", out_dir / f"fig_16{tput_panel}.pdf", max_points,
                show_yaxis=tput_yaxis,
            )


def _grid_xlim(x_col: str, band_type: str) -> Optional[tuple[float, float]]:
    if x_col == "sinr_db":
        return (-20, 40)
    if x_col == "interference_db":
        return (-140, -60)
    return None


def _grid_ylim(
    y_col: str, band_type: str, row_vals: np.ndarray
) -> Optional[tuple[float, float]]:
    if y_col == "mcs":
        return (0, 30)
    if y_col == "throughput_mbps":
        return (0, 100 if band_type == "low" else 500)
    if y_col == "prb_num":
        v = row_vals[np.isfinite(row_vals)]
        if v.size:
            hi = float(np.nanpercentile(v, 99.5)) * 1.05
            if hi > 0:
                return (0, hi)
    return None


def plot_scatter_grid(
    pop: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    out_dir: Path,
    name: str,
    max_points: int,
) -> None:
    band_types = GRID_BAND_TYPES
    x_shared = x_col in ("sinr_db", "interference_db")
    last_row = len(band_types) - 1

    fig, axes = plt.subplots(
        len(band_types),
        len(OPS),
        figsize=(5 * len(OPS), 4 * len(band_types)),
        squeeze=False,
    )

    any_data = False
    for r, band_type in enumerate(band_types):
        row_df = pop[
            (pop["band_type"] == band_type) & pop[x_col].notna() & pop[y_col].notna()
        ]
        ylim = _grid_ylim(y_col, band_type, row_df[y_col].to_numpy(dtype=float))
        xlim = _grid_xlim(x_col, band_type)
        for c, op in enumerate(OPS):
            ax = axes[r][c]
            sub = stable_sample(row_df[row_df["operator"] == op], max_points)
            if not sub.empty:
                any_data = True
                ax.scatter(
                    sub[x_col],
                    sub[y_col],
                    s=20,
                    alpha=0.75,
                    linewidths=0,
                    rasterized=True,
                )
            ax.grid(True, alpha=0.25)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.margins(x=0, y=0)
            if r == 0:
                ax.set_title(
                    OP_DISPLAY.get(op, op.upper()),
                    fontsize=LABEL_FONT_SIZE,
                    fontweight="bold",
                )
            if c == 0:
                # inner y-axis label: the metric only
                ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight="bold")
            else:
                ax.tick_params(labelleft=False)
            if r != last_row and x_shared:
                ax.tick_params(labelbottom=False)

    if not any_data:
        print(f"[WARN] No 5G rows for grid {name}.")
        plt.close(fig)
        return

    fig.supxlabel(x_label, fontsize=LABEL_FONT_SIZE, fontweight="bold")
    fig.tight_layout(rect=(0.11, 0, 1, 1), h_pad=2.0)
    for r, band_type in enumerate(band_types):
        pos = axes[r][0].get_position()
        fig.text(
            0.035,
            (pos.y0 + pos.y1) / 2.0,
            f"5G {band_type.capitalize()}",
            rotation=0,
            ha="left",
            va="center",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = {"mcs_vs_sinr": 11, "throughput_vs_sinr": 12, "prb_vs_throughput": 13}[name]
    out_path = out_dir / f"fig_{figure}.pdf"
    fig.savefig(out_path, dpi=200)
    print(f"[OK] Wrote {out_path}")
    plt.close(fig)


def plot_5g_scatter_grids(df: pd.DataFrame, out_dir: Path, max_points: int) -> None:
    plot_df = numeric_plot_frame(df)
    if plot_df.empty:
        print("[WARN] No 5G rows available for grid plotting.")
        return

    pop_all = plot_df[
        plot_df["mcs"].notna()
        & plot_df["throughput_mbps"].notna()
        & plot_df["prb_num"].notna()
    ]
    plot_scatter_grid(
        pop_all,
        "sinr_db",
        "mcs",
        "SINR (dB)",
        "DL MCS",
        out_dir,
        "mcs_vs_sinr",
        max_points,
    )
    plot_scatter_grid(
        pop_all,
        "sinr_db",
        "throughput_mbps",
        "SINR (dB)",
        "Tput (Mbps)",
        out_dir,
        "throughput_vs_sinr",
        max_points,
    )

    pop_prb = prb_tput_population(plot_df, GRID_PRB_SINR_MIN, GRID_PRB_MCS_MIN)
    plot_scatter_grid(
        pop_prb,
        "throughput_mbps",
        "prb_num",
        "Tput (Mbps)",
        "DL PRB count",
        out_dir,
        "prb_vs_throughput",
        max_points,
    )


def plot_prb_boxplot(
    df: pd.DataFrame,
    out_dir: Path,
    band_type: str,
    sinr_min_db: float,
    tput_split_mbps: float,
) -> None:
    plot_df = numeric_plot_frame(df)
    if plot_df.empty:
        print("[WARN] No 5G rows available for PRB boxplot.")
        return

    plot_df = plot_df[
        (plot_df["band_type"] == band_type)
        & plot_df["sinr_db"].notna()
        & (plot_df["sinr_db"] > sinr_min_db)
        & plot_df["throughput_mbps"].notna()
        & plot_df["prb_num"].notna()
    ]
    if plot_df.empty:
        print(
            f"[WARN] No 5G {band_type}-band rows passing "
            f"SINR>{sinr_min_db:g} dB with non-null PRB & throughput."
        )
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_YLABEL)
    positions: list[float] = []
    data: list[np.ndarray] = []
    face_colors: list[str] = []
    low_color = "#e07a5f"
    high_color = "#81b29a"

    for i, op in enumerate(OPS):
        op_df = plot_df[plot_df["operator"] == op]
        low_tput = op_df.loc[
            op_df["throughput_mbps"] < tput_split_mbps, "prb_num"
        ].to_numpy()
        high_tput = op_df.loc[
            op_df["throughput_mbps"] >= tput_split_mbps, "prb_num"
        ].to_numpy()

        positions.append(i * 2.0 + 1.0 - 0.35)
        positions.append(i * 2.0 + 1.0 + 0.35)
        data.append(low_tput)
        data.append(high_tput)
        face_colors.append(low_color)
        face_colors.append(high_color)

    keep = [i for i, arr in enumerate(data) if arr.size > 0]
    if not keep:
        print(f"[WARN] No 5G {band_type}-band PRB rows in either cohort to plot.")
        plt.close(fig)
        return

    bp = ax.boxplot(
        [data[i] for i in keep],
        positions=[positions[i] for i in keep],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch, idx in zip(bp["boxes"], keep):
        patch.set_facecolor(face_colors[idx])
        patch.set_alpha(0.75)

    ax.set_xticks([i * 2.0 + 1.0 for i in range(len(OPS))])
    ax.set_xticklabels([op.upper() for op in OPS])
    ax.set_ylabel("DL PRB count")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_xlim(0, len(OPS) * 2.0)
    ax.legend(
        handles=[
            Patch(
                facecolor=low_color,
                alpha=0.75,
                label=f"tput < {tput_split_mbps:g} Mbps",
            ),
            Patch(
                facecolor=high_color,
                alpha=0.75,
                label=f"tput ≥ {tput_split_mbps:g} Mbps",
            ),
        ],
        loc="best",
        frameon=True,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prb_5g_{band_type}.png"
    fig.savefig(out_path, dpi=200)
    out_path = out_dir / f"prb_5g_{band_type}.pdf"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def plot_prb_vs_tput_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    band_type: str,
    sinr_min_db: float,
    mcs_min: float,
    max_points: int,
    config_name: str,
) -> None:
    plot_df = numeric_plot_frame(df)
    if plot_df.empty:
        print(f"[WARN] No 5G rows for PRB-vs-tput scatter ({band_type}).")
        return

    base = prb_tput_population(plot_df, sinr_min_db, mcs_min)
    base = base[base["band_type"] == band_type]
    if base.empty:
        print(
            f"[WARN] No 5G {band_type}-band rows passing "
            f"SINR>{sinr_min_db:g} dB, MCS≥{mcs_min:g}, non-null PRB & tput."
        )
        return

    prb_vals = base["prb_num"].to_numpy(dtype=float)
    prb_vals = prb_vals[np.isfinite(prb_vals)]
    band_ylim = (
        (0.0, float(np.nanpercentile(prb_vals, 99.5)) * 1.05)
        if prb_vals.size
        else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for op in ("att", "verizon") if band_type == "mmwave" else ():
        sub = base[base["operator"] == op]
        if sub.empty:
            print(f"[WARN] No PRB-vs-tput rows for {op} 5G {band_type}.")
            continue
        original_n = len(sub)
        sub = stable_sample(sub, max_points)

        show_yaxis = band_type == "mmwave" or op == "att"
        fig, ax = plt.subplots(figsize=FIG_SIZE_YLABEL if show_yaxis else FIG_SIZE)
        ax.scatter(
            sub["throughput_mbps"],
            sub["prb_num"],
            s=24,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlabel("Tput (Mbps)")
        if show_yaxis:
            ax.set_ylabel("DL PRB count")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0, y=0)
        if band_ylim is not None:
            ax.set_ylim(*band_ylim)

        fig.tight_layout()
        out_path = out_dir / ("fig_16c.pdf" if op == "att" else "fig_16f.pdf")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Wrote {out_path}")


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = numeric_plot_frame(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot_df.empty:
        return
    summary = (
        plot_df.groupby(["operator", "band_type"], dropna=False)
        .agg(
            rows=("rat", "size"),
            mcs_rows=("mcs", lambda s: int(s.notna().sum())),
            throughput_rows=("throughput_mbps", lambda s: int(s.notna().sum())),
            median_sinr=("sinr_db", "median"),
            median_interference=("interference_db", "median"),
            median_mcs=("mcs", "median"),
            median_throughput=("throughput_mbps", "median"),
            prb_rows=("prb_num", lambda s: int(s.notna().sum())),
            median_prb=("prb_num", "median"),
        )
        .reset_index()
    )
    out_csv = out_dir / "mcs_tput_5g_plot_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--analysis-dir", type=Path, default=ROOT / "data/intermediate_data/2024")
    ap.add_argument("--operators", nargs="+", default=OPS, choices=OPS)
    ap.add_argument("--rebuild", action="store_true", help="Re-parse raw CSVs")
    ap.add_argument(
        "--correlations-only",
        action="store_true",
        help="Write PRB/throughput correlations without regenerating figures",
    )
    ap.add_argument(
        "--max-report-mb",
        type=float,
        default=50.0,
        help="Maximum decimal MB per split report file",
    )
    ap.add_argument(
        "--max-points-per-plot",
        type=int,
        default=0,
        help="Deterministic sample cap per scatter; default 0 plots all points",
    )
    args = ap.parse_args()

    report_dir = args.report_dir
    scatter_dir = args.out_dir
    analysis_dir = args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report_parts = sorted(report_dir.glob("mcs_tput_report_part*.csv"))
    if report_parts and not args.rebuild:
        print(f"[INFO] Using cached report parts under {report_dir}")
        fiveg_df = load_5g_from_reports(report_dir)
    else:
        print(f"[INFO] Parsing raw data under {args.input_root}")
        fiveg_df = process_raw_data(
            args.input_root,
            report_dir,
            args.operators,
            max_report_mb=args.max_report_mb,
        )

    write_prb_tput_correlations(fiveg_df, analysis_dir)
    if args.correlations_only:
        return
    write_summary(fiveg_df, analysis_dir)
    plot_5g_scatters(fiveg_df, scatter_dir, args.max_points_per_plot)
    plot_5g_scatter_grids(fiveg_df, scatter_dir, args.max_points_per_plot)

    for band_type, sinr_min, mcs_min in PRB_TPUT_SCATTER_CONFIGS_1:
        if band_type == "mmwave":
            plot_prb_vs_tput_scatter(
                fiveg_df, scatter_dir, band_type, sinr_min, mcs_min,
                args.max_points_per_plot, config_name="15",
            )


if __name__ == "__main__":
    main()
