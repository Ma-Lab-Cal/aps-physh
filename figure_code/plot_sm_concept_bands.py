#!/usr/bin/env python3
"""Regenerate the SM concept-bands figure (SM Fig. 1) from w1_trajectories.npz.

Concept trajectories of main-text Fig. 5 with min-max bands over the five
production runs; 2x2 panels matching the Fig. 5 grouping, three-year rolling
averages. Run after analysis/w1_trajectories.py.

Usage: python3 analysis/plot_sm_concept_bands.py [output.png]
       (default: analysis/sm_concept_bands_new.png; copy to reports/ and
        manuscript_source/ to update the SM build)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

PANELS = [
    ("(a) Atomic, nuclear & particle", range(0, 4), "upper right"),
    ("(b) Nobel-awarded concepts", range(4, 8), "upper right"),
    ("(c) Analytical vs computational", range(8, 14), "upper left"),
    ("(d) Modern concepts", range(14, 19), "upper left"),
]


def roll3(a):
    pad = np.concatenate([a[:1], a, a[-1:]], axis=0)
    return (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis" / "sm_concept_bands_new.png"
    z = np.load(ROOT / "analysis" / "w1_trajectories.npz", allow_pickle=True)
    years = z["years"]
    concepts = [str(c) for c in z["concepts"]]
    sm = np.stack([roll3(z["concept_shares"][r]) for r in range(z["concept_shares"].shape[0])])

    fig, axes = plt.subplots(2, 2, figsize=(14.22, 9.43), dpi=100)
    for ax, (title, idxs, loc) in zip(axes.flat, PANELS):
        for k, ci in enumerate(idxs):
            col = f"C{k}"
            ax.fill_between(years, sm[:, :, ci].min(0), sm[:, :, ci].max(0),
                            color=col, alpha=0.25, linewidth=0)
            ax.plot(years, sm[:, :, ci].mean(0), color=col, linewidth=1.4,
                    label=concepts[ci][:30])
        ax.set_title(title, loc="left", fontsize=16)
        ax.set_xlabel("Year", fontsize=15)
        ax.set_ylabel("Share (%)", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=10.5, loc=loc, frameon=False)
        ax.set_xlim(years.min(), years.max())
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
