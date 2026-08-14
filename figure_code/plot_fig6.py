#!/usr/bin/env python3
"""Regenerate main-text Fig. 6 (trend stability) from the released analysis artifacts.

Inputs (produced by w1_trajectories.py, w1_noise.py, build_xmodel_shares.py):
  analysis/w1_trajectories.npz : shares (5 runs x 133 years x 17 disciplines), %
  analysis/w1_noise_replicates.npz : base (133 x 17), reps (100 x 133 x 17), %
  analysis/xmodel_shares.npz : per-year shares from the independent Claude
      Sonnet 5 and Claude Haiku 4.5 relabelings of the corpus sample (counts
      grow as the relabeling program advances; see xmodel/MANIFEST.json)

Main figure: five-run min-max bands and mean plus the two cross-model trajectories
(dashed / dotted). SM figure: 5-95% bands over 100 noise replicates.
Seven major disciplines. Three-year rolling averages, with the
cross-model curves at nine-year windows after 1952 where the sample thins to 10%.

Usage: python3 analysis/plot_fig6.py [output.png]   (default: analysis/fig6_regen.png)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

def fig4_color(i):
    """Fig. 4's palette: tab20 softened exactly as figregen/fig4.py build_color_map."""
    import colorsys
    r, g, b, _ = plt.get_cmap("tab20")(i)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l * 1.1, s * 0.9)


SHOW = [  # (canonical discipline name, legend label, color = Fig. 4 palette)
    ("Condensed Matter, Materials & Applied Physics", "Condensed Matter",    fig4_color(0)),
    ("Atomic, Molecular & Optical",                   "Atomic",              fig4_color(1)),
    ("Nuclear Physics",                               "Nuclear Physics",     fig4_color(3)),
    ("Particles & Fields",                            "Particles & Fields",  fig4_color(2)),
    ("Statistical Physics & Thermodynamics",          "Statistical Physics", fig4_color(4)),
    ("Gravitation, Cosmology & Astrophysics",         "Gravitation",         fig4_color(5)),
    ("Plasma Physics",                                "Plasma Physics",      fig4_color(9)),
]


def rollk(a, k, axis=0):
    """Centered k-year rolling mean with edge replication."""
    h = k // 2
    pad = np.concatenate([np.take(a, [0] * h, axis), a, np.take(a, [-1] * h, axis)], axis)
    sl = [slice(None)] * a.ndim
    out = np.zeros_like(a, dtype=float)
    for off in range(k):
        sl[axis] = slice(off, off + a.shape[axis])
        out += pad[tuple(sl)]
    return out / k


def roll3(a, axis=0):
    return rollk(a, 3, axis)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis" / "fig6_regen.png"
    out_sm = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "analysis" / "sm_noise_panel.png"
    zt = np.load(ROOT / "analysis" / "w1_trajectories.npz", allow_pickle=True)
    zn = np.load(ROOT / "analysis" / "w1_noise_replicates.npz", allow_pickle=True)
    zx = np.load(ROOT / "analysis" / "xmodel_shares.npz", allow_pickle=True)
    years = zt["years"]
    disc = [str(d) for d in zt["disciplines"]]
    didx = {d: i for i, d in enumerate(disc)}

    shares = np.stack([roll3(zt["shares"][r]) for r in range(zt["shares"].shape[0])])
    reps = np.stack([roll3(zn["reps"][r]) for r in range(zn["reps"].shape[0])])
    # Smoothing per model, matched to sample density: 3-yr windows in the dense
    # region, 9-yr in the sparse 45/yr draw. Each region is smoothed as its own
    # segment (edge replication inside the segment) so no window mixes dense and
    # sparse years -- a cross-boundary window once pulled a noisy 45-paper year
    # into the dense curve and produced a spurious spike at the seam.
    # Sonnet: dense through 1990 (complete to 1952 + ~1,000/yr to 1990).
    # Haiku: dense through 1952 only.
    def era_split(shares_py, n_py, cut_year):
        # Dense region: plain 3-yr windows on the dense segment only (never
        # ingesting sparse years). Sparse region: 9-yr sample-weighted windows
        # over the full series, so windows near the seam are dominated by the
        # dense neighbors and single noisy 45-paper years cannot spike the curve.
        cut = np.searchsorted(years, cut_year)
        out = np.empty_like(shares_py)
        out[:cut] = rollk(shares_py[:cut], 3)
        w = n_py[:, None] * shares_py
        num = np.stack([np.convolve(w[:, j], np.ones(9), "same") for j in range(w.shape[1])], 1)
        den = np.convolve(n_py, np.ones(9), "same")
        out[cut:] = (num / den[:, None])[cut:]
        return out
    xson = era_split(zx["sonnet_shares"], zx["sonnet_n"], 1991)
    xhai = era_split(zx["haiku_shares"], zx["haiku_n"], 1971)

    # main-text Fig. 6: five-run min-max bands + independent cross-model overlay
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=100)
    handles, labels = [], []
    for name, label, col in SHOW:
        i = didx[name]
        ax.fill_between(years, shares[:, :, i].min(0), shares[:, :, i].max(0),
                        color=col, alpha=0.35, linewidth=0)
        ax.plot(years, shares[:, :, i].mean(0), color=col, lw=1.0)
        ax.plot(years, xson[:, i], color=col, lw=1.5, ls="--", alpha=0.9)
        ax.plot(years, xhai[:, i], color=col, lw=1.1, ls=":", alpha=0.75)
        handles.append(Line2D([], [], color=col, lw=2.2))
        labels.append(label)
    handles += [Line2D([], [], color="k", lw=1.5, ls="--"),
                Line2D([], [], color="k", lw=1.1, ls=":")]
    labels += ["Sonnet 5", "Haiku 4.5"]
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 62)
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Share of papers (%)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(handles, labels, fontsize=9, ncol=3, frameon=False,
              loc="upper right", handlelength=1.3, columnspacing=0.9,
              labelspacing=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")

    # SM figure: stationary-error noise injection (replicate median + 5-95% band);
    # the level shift relative to the as-labeled curves is the systematic effect
    # of the precision-weighted replacement, discussed in SM Sec. S2
    fig2, ax2 = plt.subplots(figsize=(7.4, 5.43), dpi=100)
    for name, label, col in SHOW:
        i = didx[name]
        lo = np.percentile(reps[:, :, i], 5, axis=0)
        hi = np.percentile(reps[:, :, i], 95, axis=0)
        med = np.percentile(reps[:, :, i], 50, axis=0)
        ax2.fill_between(years, lo, hi, color=col, alpha=0.35, linewidth=0)
        ax2.plot(years, med, color=col, lw=1.5, label=label)
    ax2.set_xlim(years.min(), years.max())
    ax2.set_ylim(0, 60)
    ax2.set_xlabel("Year", fontsize=16)
    ax2.set_ylabel("Share of papers (%)", fontsize=16)
    ax2.tick_params(labelsize=14)
    ax2.legend(fontsize=11, ncol=2, frameon=False, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_sm)
    print(f"wrote {out_sm}")


if __name__ == "__main__":
    main()
