#!/usr/bin/env python3
"""Build the Fig. 1(a) coverage cache from released 2022 measurements.

The two output dictionaries contain total route miles and miles by technology.
The public measurement CSV keeps the columns needed here and replaces device
phone numbers with the same aliases used elsewhere in this dataset.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import pickle

from geopy.distance import geodesic
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "data/raw_data/2022/coverage/coverage_measurements.csv"
GPS = ROOT / "data/raw_data/2022/coverage/atnt_gps_fill.csv"
STATIC = ROOT / "data/raw_data/2022/coverage/static"
OUTPUT = ROOT / "data/intermediate_data/2022/coverage/dist_tz_speed_operator.pkl"

FIELDS = [
    "TIME_STAMP", "Lat", "Lon", "Event Technology",
    "Qualcomm Lte/LteAdv Intrafreq Measure PCell Frequency(DL)[MHz]",
    "5G KPI PCell RF Frequency [MHz]",
]
PHONE = "Smart Phone Android Mobile Info Line1 Number"
OPERATOR = "Smart Phone Android System Info Operator"
PHONE_TO_OP = {
    "device_01": "verizon", "device_02": "verizon", "device_03": "verizon",
    "device_06": "tmobile", "device_07": "tmobile", "device_08": "tmobile",
    "device_04": "atnt", "device_05": "atnt",
}
OP_LABEL = {
    "Verizon ": "verizon", "AT&T": "atnt", "No service": "atnt",
    "T-Mobile": "tmobile", "Viaero Wireless": "tmobile", "(Searching)": "tmobile",
}
REPRESENTATIVE_PHONE = {
    "verizon": "device_01", "tmobile": "device_06", "atnt": "device_04",
}
TECHNOLOGIES = (
    "5G-sub6", "LTE-A", "5G-mmWave 28 GHz", "5G-low", "LTE",
    "5G-mmWave 39 GHz",
)


def separate_operators() -> dict[str, pd.DataFrame]:
    """Reproduce the old operator state machine without its large pickle."""
    print("Reading combined measurements", flush=True)
    df = pd.read_csv(COMBINED, usecols=FIELDS + [PHONE, OPERATOR], low_memory=False)
    df = df.iloc[5:].reset_index(drop=True)  # original parser skipped five rows
    current_phone = df.at[0, PHONE]
    current_op = PHONE_TO_OP[current_phone]
    codes = {"verizon": 0, "tmobile": 1, "atnt": 2}
    assignment = np.empty(len(df), dtype=np.int8)
    assignment[0] = codes[current_op]
    last_transition = 0
    transitions = defaultdict(int)

    for i, (phone, label) in enumerate(
        zip(df[PHONE].array[1:], df[OPERATOR].array[1:]), start=1
    ):
        next_phone = current_phone
        next_op = current_op
        changed = False
        if pd.notna(phone):
            if phone == "unknown":
                if current_op != "tmobile":
                    next_phone = REPRESENTATIVE_PHONE["tmobile"]
                    next_op = "tmobile"
                    changed = True
            else:
                candidate = str(phone)
                if candidate != current_phone:
                    next_phone = candidate
                    next_op = PHONE_TO_OP[candidate]
                    changed = True
        elif pd.notna(label):
            candidate_op = OP_LABEL.get(label)
            if candidate_op is not None and candidate_op != current_op:
                next_op = candidate_op
                next_phone = REPRESENTATIVE_PHONE[candidate_op]
                changed = True
        if changed:
            last_transition = i
            transitions[current_op] += 1
            current_op, current_phone = next_op, next_phone
        assignment[i] = codes[current_op]

    # The old script never flushed its last segment, even when it had rows.
    omitted = len(df) - last_transition
    df = df.iloc[:last_transition]
    assignment = assignment[:last_transition]
    result = {op: df.loc[assignment == code, FIELDS].copy()
              for op, code in codes.items()}
    print("Operator rows:", {op: len(part) for op, part in result.items()}, flush=True)
    print("Omitted final segment rows:", omitted, flush=True)
    return result


def fill_atnt_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Copy modify_atnt_df's nearest-timestamp join and 400-row trim."""
    print("Filling missing AT&T coordinates", flush=True)
    gps = pd.read_csv(GPS, usecols=["TIME_STAMP", "Lat", "Lon"], low_memory=False)
    gps = gps.iloc[:-400].sort_values("TIME_STAMP")
    good = df.loc[df["Lat"].notna() & df["Lon"].notna()]
    missing = df.loc[df["Lat"].isna() & df["Lon"].isna()]
    missing = missing.drop(columns=["Lat", "Lon"]).sort_values("TIME_STAMP")
    merged = pd.merge_asof(
        missing, gps, on="TIME_STAMP", direction="nearest", tolerance=300
    )
    result = pd.concat([good, merged]).sort_values("TIME_STAMP")
    print(f"AT&T missing GPS rows: {len(missing)}; filled: "
          f"{merged['Lat'].notna().sum()}", flush=True)
    return result


def static_intervals(op: str, timezone: ZoneInfo) -> list[tuple[int, int]]:
    starts = []
    for path in (STATIC / op).glob("*.csv"):
        stamp = pd.read_csv(path, usecols=["TIME_STAMP"], nrows=3).iloc[2, 0]
        local = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone)
        starts.append(int(local.timestamp()))
    intervals = []
    for stamp in sorted(starts):
        if not intervals or not intervals[-1][0] <= stamp < intervals[-1][1]:
            intervals.append((stamp, stamp + 3600))
    return intervals


def mark_static(df: pd.DataFrame, op: str, timezone: ZoneInfo, *, retain: bool) -> pd.DataFrame:
    stamp = df["TIME_STAMP"].astype(int)
    mask = np.zeros(len(df), dtype=bool)
    for start, end in static_intervals(op, timezone):
        mask |= ((stamp >= start) & (stamp < end)).to_numpy()
    print(f"{op} static rows: {int(mask.sum())}", flush=True)
    if retain:
        df = df.copy()
        df["TIME_STAMP"] = stamp.astype(object)
        df.loc[mask, "TIME_STAMP"] = "static"
        return df
    return df.loc[~mask].assign(TIME_STAMP=stamp.loc[~mask])


def classify(frequency: float, nearby: list[float], *, nr: bool) -> str:
    if int(frequency) == 0:
        nonzero = [value for value in nearby if value != 0]
        if not nonzero:
            return ""
        frequency = np.mean(nonzero)
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


def distance_by_technology(df: pd.DataFrame, op: str) -> tuple[float, dict[str, float]]:
    # Original code fills missing values with zero before frequency lookup.
    df = df[FIELDS].fillna(0)
    ts, lat, lon, events, lte, nr = (df[column].to_numpy() for column in FIELDS)
    unique = {}
    for i, (stamp, y, x, event) in enumerate(zip(ts, lat, lon, events)):
        if event == 0 or event == "0" or event == "0.0":
            continue
        if "5G" in event:
            freq = nr
            is_nr = True
        elif "LTE" in event:
            freq = lte
            is_nr = False
        else:
            continue
        near = freq[i - 10:i] if i > len(freq) - 10 else freq[i + 1:i + 10]
        tech = classify(freq[i], near, nr=is_nr)
        if not tech:
            continue
        if op == "verizon" and "39 GHz" in tech and stamp in range(1660432274, 1660433472):
            continue
        if op == "tmobile" and "mmWave" in tech and stamp > 1659983687:
            continue
        if stamp == "static":
            continue
        unique.setdefault((stamp, y, x), tech)

    points = defaultdict(list)
    for key, tech in sorted(unique.items()):
        points[tech].append(key)
    total = 0.0
    breakdown = dict.fromkeys(TECHNOLOGIES, 0.0)
    for tech, samples in points.items():
        previous = samples[0]
        for current in samples[1:]:
            if current[0] - previous[0] <= 5:
                miles = geodesic(current[1:], previous[1:]).miles
                if miles <= 0.3:
                    total += miles
                    breakdown[tech] += miles
            previous = current
    print(f"{op}: {len(unique)} distinct samples, {total:.6f} miles", flush=True)
    return total, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-timezone", default="America/New_York",
                        help="timezone used to interpret the old static CSV timestamps")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    timezone = ZoneInfo(args.static_timezone)
    print("Static log timezone:", timezone, flush=True)
    frames = separate_operators()
    frames["atnt"] = fill_atnt_gps(frames["atnt"])
    totals = {}
    breakdowns = {}
    for op in ("verizon", "tmobile", "atnt"):
        frame = mark_static(frames[op], op, timezone, retain=(op == "atnt"))
        totals[op], breakdowns[op] = distance_by_technology(frame, op)
        del frame, frames[op]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump((totals, breakdowns), handle)
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
