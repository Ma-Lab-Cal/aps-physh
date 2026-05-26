from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from physh_tools.io import paper_doi, paper_year

SMOOTH_WINDOW = 3
PLOT_AREA_W = 3.27
PLOT_AREA_H = 1.88
LEFT_MARGIN = 0.50
RIGHT_MARGIN = 0.52
BOTTOM_MARGIN = 0.34
TOP_MARGIN = 0.08
LABEL_FONTSIZE = 7.5
TICK_FONTSIZE = 7.5
SPINE_LW = 0.7
TICK_LW = 0.6


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_years(metadata_path: Path | None) -> dict[str, int]:
    years: dict[str, int] = {}
    if metadata_path is None:
        return years
    for row in iter_jsonl(metadata_path):
        doi = paper_doi(row)
        year = paper_year(row)
        if doi and year:
            years[doi] = int(year)
    return years


def aggregate_runs(
    run_paths: list[Path],
    *,
    years_by_doi: dict[str, int],
    ordered_disciplines: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    valid_disciplines = set(ordered_disciplines)
    run_frames = []

    for run_path in run_paths:
        counts: dict[int, Counter[str]] = defaultdict(Counter)
        for row in iter_jsonl(run_path):
            doi = paper_doi(row)
            year_text = paper_year(row)
            year = int(year_text) if year_text else years_by_doi.get(doi)
            if year is None or year < start_year or year > end_year:
                continue
            disciplines = [d for d in dict.fromkeys(row.get("predicted_disciplines", [])) if d in valid_disciplines]
            if not disciplines:
                continue
            weight = 1.0 / len(disciplines)
            for discipline in disciplines:
                counts[year][discipline] += weight

        years = range(start_year, end_year + 1)
        run_frames.append(
            pd.DataFrame(
                [
                    {discipline: counts[year].get(discipline, 0.0) for discipline in ordered_disciplines}
                    for year in years
                ],
                index=list(years),
            )
        )

    if not run_frames:
        raise SystemExit("No run files were provided.")
    return sum(run_frames) / len(run_frames)


def smoothed_counts(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()


def configure_matplotlib(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "font.size": 6,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "axes.linewidth": SPINE_LW,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_fractional_share(
    counts: pd.DataFrame,
    ordered_disciplines: list[str],
    color_map: dict[str, str],
    output: Path,
    *,
    dpi: int,
) -> None:
    configure_matplotlib(dpi)
    total_counts = counts.sum(axis=1)
    shares = counts.div(total_counts.replace(0, np.nan), axis=0).fillna(0.0)
    years = shares.index.to_numpy()

    fig_w = LEFT_MARGIN + PLOT_AREA_W + RIGHT_MARGIN
    fig_h = BOTTOM_MARGIN + PLOT_AREA_H + TOP_MARGIN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [
            LEFT_MARGIN / fig_w,
            BOTTOM_MARGIN / fig_h,
            PLOT_AREA_W / fig_w,
            PLOT_AREA_H / fig_h,
        ]
    )

    bottom = np.zeros(len(years))
    for discipline in ordered_disciplines:
        heights = shares[discipline].values
        ax.bar(
            years,
            heights,
            bottom=bottom,
            width=1.0,
            color=color_map[discipline],
            edgecolor="none",
            align="edge",
            linewidth=0,
        )
        bottom += heights

    ax.set_ylabel("Share of papers (%)", fontsize=LABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol=""))
    ax.set_ylim(0, 1.0)
    ax.set_xlim(years.min(), years.max())
    ax.set_xticks([tick for tick in [1900, 1920, 1940, 1960, 1980, 2000, 2020] if years.min() <= tick <= years.max() + 1])
    ax.set_xlabel("Year", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", width=TICK_LW, length=2.2, pad=1.5, labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_split_counts(
    counts: pd.DataFrame,
    ordered_disciplines: list[str],
    color_map: dict[str, str],
    output: Path,
    *,
    dpi: int,
    split_year: int,
) -> None:
    configure_matplotlib(dpi)
    years = counts.index.to_numpy()
    counts_thousands = (counts[ordered_disciplines] / 1000.0).to_numpy()

    fig_w = LEFT_MARGIN + PLOT_AREA_W + RIGHT_MARGIN
    fig_h = BOTTOM_MARGIN + PLOT_AREA_H + TOP_MARGIN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    plot_x0 = LEFT_MARGIN / fig_w
    plot_y0 = BOTTOM_MARGIN / fig_h
    plot_w = PLOT_AREA_W / fig_w
    plot_h = PLOT_AREA_H / fig_h
    left_w = plot_w * 0.5
    right_w = plot_w * 0.5

    ax_left = fig.add_axes([plot_x0, plot_y0, left_w, plot_h])
    ax_right = fig.add_axes([plot_x0 + left_w, plot_y0, right_w, plot_h])

    def draw_stacked_bars(ax):
        bottom = np.zeros(len(years))
        for index, discipline in enumerate(ordered_disciplines):
            ax.bar(
                years,
                counts_thousands[:, index],
                bottom=bottom,
                width=1.0,
                color=color_map[discipline],
                alpha=0.88,
                edgecolor="none",
                align="center",
                linewidth=0,
            )
            bottom += counts_thousands[:, index]

    draw_stacked_bars(ax_left)
    draw_stacked_bars(ax_right)

    ax_left.set_xlim(years.min(), split_year)
    ax_right.set_xlim(split_year, years.max())
    ax_left.set_ylim(0, 2.2)
    ax_right.set_ylim(0, 38.0)
    ax_left.yaxis.set_major_formatter(mtick.FormatStrFormatter("%g"))
    ax_right.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_left.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_right.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax_left.set_xticks([1900, 1920, 1940])
    ax_right.set_xticks([1980, 2000, 2020])

    for ax in (ax_left, ax_right):
        ax.tick_params(axis="x", width=TICK_LW, length=2.2, pad=1.5, labelsize=TICK_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
    ax_left.tick_params(axis="y", left=True, labelleft=True, right=False, width=TICK_LW, length=2.2, pad=1.5, labelsize=TICK_FONTSIZE)
    ax_right.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, width=TICK_LW, length=2.2, pad=1.5, labelsize=TICK_FONTSIZE)
    ax_left.set_ylabel("No. of papers (in thousands)", fontsize=LABEL_FONTSIZE, labelpad=2)
    fig.text(plot_x0 + plot_w / 2, plot_y0 - 0.24 / fig_h, "Year", ha="center", va="top", fontsize=LABEL_FONTSIZE)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    seam_x_fig = plot_x0 + left_w
    divider = plt.Line2D(
        [seam_x_fig, seam_x_fig],
        [plot_y0, plot_y0 + plot_h],
        transform=fig.transFigure,
        color="0.25",
        linewidth=0.9,
        linestyle=(0, (2.0, 2.0)),
        zorder=10,
    )
    fig.add_artist(divider)
    fig.text(seam_x_fig, plot_y0 - 0.055 / fig_h, str(split_year), ha="center", va="top", fontsize=LABEL_FONTSIZE)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_legend(ordered_disciplines: list[str], color_map: dict[str, str], output: Path, *, dpi: int) -> None:
    short_names = {
        "Condensed Matter, Materials & Applied Physics": "Cond. Matter, Materials & Appl. Phys.",
        "Gravitation, Cosmology & Astrophysics": "Gravitation, Cosmology & Astrophys.",
        "Quantum Information, Science & Technology": "Quantum Info., Science & Tech.",
    }
    fig_h = BOTTOM_MARGIN + PLOT_AREA_H + TOP_MARGIN
    fig = plt.figure(figsize=(4.0, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[discipline]) for discipline in ordered_disciplines]
    labels = [short_names.get(discipline, discipline) for discipline in ordered_disciplines]
    ax.legend(
        list(reversed(handles)),
        list(reversed(labels)),
        loc="center",
        frameon=False,
        fontsize=6,
        ncols=2,
        columnspacing=1.5,
        handlelength=1.2,
        handleheight=1.2,
        labelspacing=0.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 4-style discipline trends from classification runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run JSONL/JSONL.GZ files.")
    parser.add_argument("--metadata", required=True, help="JSONL file with DOI and year fields.")
    parser.add_argument("--output-share", default="outputs/fig_discipline_share.png")
    parser.add_argument("--output-counts", default="outputs/fig_discipline_counts_split.png")
    parser.add_argument("--output-legend", default="outputs/fig_discipline_legend.png")
    parser.add_argument("--start-year", type=int, default=1893)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--split-year", type=int, default=1960)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    color_map = load_json(ROOT / "style" / "discipline_colors.json")
    ordered_disciplines = list(color_map.keys())
    years_by_doi = load_years(Path(args.metadata))
    counts = aggregate_runs(
        [Path(path) for path in args.runs],
        years_by_doi=years_by_doi,
        ordered_disciplines=ordered_disciplines,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    counts = smoothed_counts(counts)
    plot_fractional_share(counts, ordered_disciplines, color_map, Path(args.output_share), dpi=args.dpi)
    plot_split_counts(counts, ordered_disciplines, color_map, Path(args.output_counts), dpi=args.dpi, split_year=args.split_year)
    plot_legend(ordered_disciplines, color_map, Path(args.output_legend), dpi=args.dpi)
    print(f"Wrote discipline share figure -> {args.output_share}")
    print(f"Wrote discipline count figure -> {args.output_counts}")
    print(f"Wrote discipline legend -> {args.output_legend}")


if __name__ == "__main__":
    main()
