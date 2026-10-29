#!/usr/bin/env python3
"""Build the Fig. 1(b) coverage cache from released 2024 XCAL exports.

The two output dictionaries contain total route miles and miles by technology.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import pickle

from earfcn.convert import earfcn2freq
from geopy.distance import geodesic
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_data/2024/xcal"
AREA = ROOT / "data/raw_data/2024/area.csv"
OUTPUT = ROOT / "data/intermediate_data/2024/coverage/tz_dist_coverage.pkl"

FILES = {"verizon": "V", "tmobile": "T", "atnt": "A"}
TECHS = ("LTE", "LTE-A", "5G-low", "5G-sub6", "5G-mmWave 28 GHz", "5G-mmWave 39 GHz")
COLUMNS = [
    "TIME_STAMP", "Unix TIME_STAMP", "Lat", "Lon", "Event Technology",
    "LTE KPI PCell Serving EARFCN(DL)", "5G KPI PCell RF Frequency [MHz]",
]


def lte_frequency(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    def convert(earfcn: float) -> float:
        try:
            return earfcn2freq(earfcn)
        except Exception:
            return np.nan
    lookup = {value: convert(value) for value in values.dropna().unique()}
    return values.map(lookup)


def tech_label(frequency: float, nearby: np.ndarray, *, nr: bool) -> str:
    if int(frequency) == 0:
        nonzero = nearby[nearby != 0]
        if not len(nonzero):
            return ""
        frequency = float(np.mean(nonzero))
    if nr:
        if frequency < 1000:
            return "5G-low"
        if 1000 < frequency < 7000:
            return "5G-sub6"
        if 7000 < frequency < 35000:
            return "5G-mmWave 28 GHz"
        if frequency > 35000:
            return "5G-mmWave 39 GHz"
    else:
        if frequency < 1000:
            return "LTE"
        if frequency > 1000:
            return "LTE-A"
    return ""


def read_day(path: Path, area: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_excel(path, engine="calamine", usecols=COLUMNS)
    df = df.iloc[:-8].copy()  # legacy script discards eight trailing rows
    timestamps = pd.to_datetime(df["TIME_STAMP"], errors="coerce")
    unix = pd.to_numeric(df["Unix TIME_STAMP"], errors="coerce")
    df["TIME_STAMP"] = unix + timestamps.dt.microsecond.fillna(0) / 1_000_000
    if df["TIME_STAMP"].isna().any():
        raise ValueError(f"Missing Unix timestamp in {path}")
    df["LTE_FREQ"] = lte_frequency(df["LTE KPI PCell Serving EARFCN(DL)"])
    fields = ["TIME_STAMP", "Lat", "Lon", "Event Technology", "LTE_FREQ",
              "5G KPI PCell RF Frequency [MHz]"]
    measurements = df[fields]
    lower = measurements["TIME_STAMP"].iloc[0] - 3600
    upper = measurements["TIME_STAMP"].iloc[-1]
    markers = area.loc[(area["TIME_STAMP"] >= lower) &
                       (area["TIME_STAMP"] <= upper), ["TIME_STAMP", "value"]]
    joined = pd.concat([measurements, markers], ignore_index=True)
    joined = joined.sort_values("TIME_STAMP")
    joined["value"] = joined["value"].ffill().fillna(0)
    return joined.fillna(0)


def accumulate(unique: dict, df: pd.DataFrame) -> None:
    ts = df["TIME_STAMP"].to_numpy()
    lat = df["Lat"].to_numpy()
    lon = df["Lon"].to_numpy()
    events = df["Event Technology"].to_numpy()
    lte = df["LTE_FREQ"].to_numpy()
    nr = df["5G KPI PCell RF Frequency [MHz]"].to_numpy()
    areas = df["value"].to_numpy()
    n = len(df)
    for i, event in enumerate(events):
        if not isinstance(event, str) or event in ("0", "0.0"):
            continue
        if "5G" in event:
            freq, is_nr = nr, True
        elif "LTE" in event:
            freq, is_nr = lte, False
        else:
            continue
        near = freq[i - 10:i] if i > n - 10 else freq[i + 1:i + 10]
        tech = tech_label(freq[i], near, nr=is_nr)
        if tech:
            unique.setdefault((ts[i], lat[i], lon[i], areas[i]), tech)


def distance_summary(unique: dict) -> tuple[float, dict[str, float]]:
    points = defaultdict(list)
    for key, tech in sorted(unique.items()):
        points[tech].append(key)
    total = 0.0
    breakdown = dict.fromkeys(TECHS, 0.0)
    for tech, samples in points.items():
        previous = samples[0]
        for current in samples[1:]:
            dt = current[0] - previous[0]
            if 0 <= dt <= 5:
                miles = geodesic(current[1:3], previous[1:3]).miles
                if miles <= 0.3:
                    total += miles
                    breakdown[tech] += miles
            previous = current
    return total, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    area = pd.read_csv(AREA, usecols=["utc_ts", "value"])
    area = area.rename(columns={"utc_ts": "TIME_STAMP"})
    totals, breakdowns = {}, {}
    for op, prefix in FILES.items():
        unique = {}
        for day in range(1, 9):
            path = INPUT / f"{prefix}_{day}_100MS.xlsx"
            df = read_day(path, area)
            accumulate(unique, df)
            print(f"{op} day {day}: {len(df)} rows, {len(unique)} unique samples", flush=True)
            del df
        totals[op], breakdowns[op] = distance_summary(unique)
        print(f"{op}: {totals[op]:.6f} miles", flush=True)
        del unique

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump((totals, breakdowns), handle)
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
