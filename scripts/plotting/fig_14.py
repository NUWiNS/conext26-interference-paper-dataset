#!/usr/bin/env python3
"""Plot the 2024 cross-country driving route from one operator's XCAL logs"""

from pathlib import Path
import argparse
import importlib.util
from collections import Counter

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
XCAL_DIR = ROOT / "data/raw_data/2024/xcal"
DEFAULT_OUTPUT = ROOT / "plots/fig_14.pdf"
OPERATOR_PREFIX = {"att": "A", "tmobile": "T", "verizon": "V"}


def read_route(operator: str) -> pd.DataFrame:
    files = sorted(XCAL_DIR.glob(f"{OPERATOR_PREFIX[operator]}_*_100MS.xlsx"))
    if not files:
        raise FileNotFoundError(f"No {operator} XCAL files found in {XCAL_DIR}")

    engine = "calamine" if importlib.util.find_spec("python_calamine") else None
    parts = []
    for path in files:
        points = pd.read_excel(path, usecols=["Lat", "Lon"], engine=engine)
        points = points.apply(pd.to_numeric, errors="coerce").dropna()
        points = points.loc[
            points["Lat"].between(24, 50) & points["Lon"].between(-125, -66)
        ]

        parts.append(points.round(3).drop_duplicates())
        print(f"{path.name}: {len(parts[-1]):,} map positions")

    route = pd.concat(parts, ignore_index=True).drop_duplicates()
    if route.empty:
        raise ValueError(f"No valid GPS positions found for {operator}")

    # Discard isolated GPS glitches
    tiles = list(
        zip(
            (route["Lat"] / 0.02).floordiv(1).astype(int),
            (route["Lon"] / 0.02).floordiv(1).astype(int),
        )
    )
    tile_counts = Counter(tiles)
    supported = {
        tile: sum(
            tile_counts[(tile[0] + dlat, tile[1] + dlon)]
            for dlat in (-1, 0, 1)
            for dlon in (-1, 0, 1)
        ) >= 5
        for tile in tile_counts
    }
    before = len(route)
    route = route.loc[[supported[tile] for tile in tiles]]
    print(f"Discarded {before - len(route):,} isolated GPS positions")
    return route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=OPERATOR_PREFIX, default="att")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    route = read_route(args.operator)
    fig = go.Figure(
        go.Scattergeo(
            lat=route["Lat"],
            lon=route["Lon"],
            mode="markers",
            marker={"color": "black", "size":5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_geos(scope="usa", projection_type="albers usa")
    fig.update_layout(
        width=1100,
        height=700,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(args.output))
    print(f"Wrote {args.output} from {len(route):,} map positions")


if __name__ == "__main__":
    main()
