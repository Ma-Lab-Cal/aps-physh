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
import pandas as pd

from physh_tools.io import paper_doi, paper_year


FIGURE5_PANELS = [
    {
        "series": [
            ("Atomic spectra", ["Atomic spectra"]),
            ("Nuclear reactions", ["Nuclear reactions"]),
            ("Mesons", ["Mesons"]),
            ("Quantum chromodynamics", ["Quantum chromodynamics"]),
        ],
    },
    {
        "series": [
            ("X-ray diffraction", ["X-ray diffraction"]),
            ("Nuclear magnetic resonance", ["Nuclear magnetic resonance"]),
            ("Cosmic microwave background", ["Cosmic microwave background"]),
            ("Graphene", ["Graphene"]),
        ],
        "discovery_years": {
            "X-ray diffraction": 1912,
            "Nuclear magnetic resonance": 1946,
            "Cosmic microwave background": 1965,
            "Graphene": 2004,
        },
        "nobel_years": {
            "X-ray diffraction": 1914,
            "Nuclear magnetic resonance": 1952,
            "Cosmic microwave background": 1978,
            "Graphene": 2010,
        },
    },
    {
        "series": [
            (
                "Analytical frameworks",
                ["Mathematical physics", "Perturbation theory", "Green's function methods"],
            ),
            (
                "Computational methods",
                ["Density functional theory", "Monte Carlo methods", "Tensor network methods"],
            ),
        ],
    },
    {
        "series": [
            ("Topological insulators", ["Topological insulators"]),
            ("Quantum information processing", ["Quantum information processing"]),
            ("Machine learning", ["Machine learning"]),
            ("Gravitational wave sources", ["Gravitational wave sources"]),
            ("Spintronics", ["Spintronics"]),
        ],
    },
]

PANEL_COLORS = [
    ["#4C78A8", "#F58518", "#54A24B", "#B279A2"],
    ["#4C78A8", "#F58518", "#54A24B", "#E45756"],
    ["#4C78A8", "#E45756"],
    ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"],
]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    min_run_count: int,
    start_year: int | None,
    end_year: int | None,
) -> tuple[dict[str, dict[int, int]], dict[int, int]]:
    per_paper_concepts: dict[str, Counter[str]] = defaultdict(Counter)
    paper_years: dict[str, int] = {}

    for run_path in run_paths:
        for row in iter_jsonl(run_path):
            doi = paper_doi(row)
            if not doi:
                continue
            year_text = paper_year(row)
            year = int(year_text) if year_text else years_by_doi.get(doi)
            if year is None:
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            paper_years[doi] = year
            for concept in set(row.get("predicted_concepts", [])):
                per_paper_concepts[doi][concept] += 1

    year_totals: dict[int, int] = defaultdict(int)
    concept_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for doi, year in paper_years.items():
        year_totals[year] += 1
        for concept, count in per_paper_concepts[doi].items():
            if count >= min_run_count:
                concept_counts[concept][year] += 1

    return concept_counts, dict(year_totals)


def lighten(hex_color: str, amount: float = 0.55) -> tuple[float, float, float]:
    r, g, b = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (
        min(1.0, r + (1.0 - r) * amount),
        min(1.0, g + (1.0 - g) * amount),
        min(1.0, b + (1.0 - b) * amount),
    )


def series_share(
    concepts: list[str],
    concept_counts: dict[str, dict[int, int]],
    year_totals: dict[int, int],
    years: list[int],
    smooth_years: int,
) -> pd.Series:
    values = []
    for year in years:
        total = year_totals.get(year, 0)
        count = sum(concept_counts.get(concept, {}).get(year, 0) for concept in concepts)
        values.append((100.0 * count / total) if total else 0.0)
    series = pd.Series(values, index=years)
    if smooth_years > 1:
        series = series.rolling(smooth_years, center=True, min_periods=1).mean()
    return series


def plot_panels(
    panels: list[dict],
    concept_counts: dict[str, dict[int, int]],
    year_totals: dict[int, int],
    *,
    smooth_years: int,
    output: Path,
    show: bool,
) -> None:
    years = sorted(year_totals)
    if not years:
        raise SystemExit("No papers with usable years were found.")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    axes = axes.flatten()

    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        colors = PANEL_COLORS[panel_index]
        label_to_color = {}
        for series_index, (label, concepts) in enumerate(panel["series"]):
            color = colors[series_index % len(colors)]
            label_to_color[label] = color
            series = series_share(concepts, concept_counts, year_totals, years, smooth_years)
            ax.plot(series.index, series.values, linewidth=1.9, label=label, color=color)

        for label, year in panel.get("discovery_years", {}).items():
            if label in label_to_color:
                ax.axvline(year, color=lighten(label_to_color[label]), linewidth=1.0, alpha=0.75)
        for label, year in panel.get("nobel_years", {}).items():
            if label in label_to_color:
                ax.axvline(
                    year,
                    color=lighten(label_to_color[label]),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.75,
                )

        ax.set_ylabel("Share of papers (%)")
        ax.legend(loc="upper left", fontsize=7.3, handlelength=1.5, borderaxespad=0.2)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8, length=3)

    for ax in axes[-2:]:
        ax.set_xlabel("Year")
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figure 5-style PhySH concept trajectories from classification runs."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run JSONL/JSONL.GZ files.")
    parser.add_argument("--metadata", help="Optional JSONL file with DOI and year fields.")
    parser.add_argument("--output", default="outputs/fig_concept_trends.png")
    parser.add_argument("--min-run-count", type=int, default=2, help="Minimum runs that must assign a concept.")
    parser.add_argument("--smooth-years", type=int, default=3)
    parser.add_argument("--start-year", type=int, default=1900)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_paths = [Path(path) for path in args.runs]
    missing = [str(path) for path in run_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Run file(s) not found: {', '.join(missing)}")

    years = load_years(Path(args.metadata)) if args.metadata else {}
    concept_counts, year_totals = aggregate_runs(
        run_paths,
        years_by_doi=years,
        min_run_count=args.min_run_count,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    plot_panels(
        FIGURE5_PANELS,
        concept_counts,
        year_totals,
        smooth_years=args.smooth_years,
        output=Path(args.output),
        show=args.show,
    )
    print(f"Wrote concept trajectory figure -> {args.output}")


if __name__ == "__main__":
    main()
