#!/usr/bin/env python3
""" 2022 coverage summary."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/intermediate_data/2022/coverage/dist_tz_speed_operator.pkl"
OUTPUT = ROOT / "plots/fig_1a.pdf"
OPERATORS = ("verizon", "tmobile", "atnt")
LABELS = ("Verizon", "T-Mobile", "AT&T")
TECHNOLOGIES = (
    "LTE", "5G-low", "5G-sub6", "5G-mmWave 28 GHz", "5G-mmWave 39 GHz"
)
COLORS = {
    "LTE": "#08710C",
    "5G-low": "#F3FF33",
    "5G-sub6": "#FFB233",
    "5G-mmWave 28 GHz": "#FF4629",
    "5G-mmWave 39 GHz": "#CB0404",
}


def main() -> None:
    with INPUT.open("rb") as handle:
        total_dist, breakup_dist, *_ = pickle.load(handle)

    # separate runs
    breakup_dist["atnt"]["5G-mmWave 39 GHz"] += 0.5
    fractions = {}
    for tech in TECHNOLOGIES:
        fractions[tech] = [
            100 * (
                breakup_dist[op]["LTE"] + breakup_dist[op]["LTE-A"]
                if tech == "LTE" else breakup_dist[op][tech]
            ) / total_dist[op]
            for op in OPERATORS
        ]

    fig, ax = plt.subplots(figsize=(17, 6))
    left = [0.0] * len(OPERATORS)
    for tech in TECHNOLOGIES:
        ax.barh(LABELS, fractions[tech], 0.35, left=left, color=COLORS[tech])
        left = [a + b for a, b in zip(left, fractions[tech])]
    ax.set_xlabel("Fraction of miles covered (%)", fontsize=30, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.invert_yaxis()
    ax.set_ylim(len(LABELS) - 0.5, -1.35)
    ax.tick_params(axis="both", labelsize=30, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.legend(
        [Line2D([0], [0], color=COLORS[tech], lw=6) for tech in TECHNOLOGIES[:3]],
        ["LTE", "5G-low", "5G-mid"],
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncols=3,
        columnspacing=1.1, handlelength=1.6, borderaxespad=0.2,
        prop={"size": 25, "weight": "bold"},
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
