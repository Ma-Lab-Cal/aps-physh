from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

YMIN = 10
BASE_YMAX = 50000
DEFAULT_FIGSIZE = (3.0, 1.55)
BAR_WIDTH = 1.0
BAR_EDGE_COLOR = "white"
BAR_EDGE_LW = 0.18
PACS_START_YEAR = 1970
PACS_END_YEAR = 2016
PACS_TRANSITION_END_YEAR = 1975
RESEARCH_ARTICLE_TYPES = {"article", "rapid", "brief"}

def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def paper_year(record: dict) -> int | None:
    date = record.get("date") or record.get("published") or record.get("publicationDate")
    if isinstance(date, str) and len(date) >= 4:
        try:
            return int(date[:4])
        except ValueError:
            pass
    rights = record.get("rights")
    if isinstance(rights, dict) and rights.get("copyrightYear"):
        return int(rights["copyrightYear"])
    if record.get("year"):
        return int(record["year"])
    return None


def is_research_article(record: dict) -> bool:
    return record.get("articleType") in RESEARCH_ARTICLE_TYPES


def has_physh(record: dict) -> bool:
    physh_top = record.get("physh")
    if isinstance(physh_top, list) and physh_top:
        return True
    schemes = record.get("classificationSchemes")
    physh_obj = schemes.get("physh") if isinstance(schemes, dict) else None
    if isinstance(physh_obj, dict):
        concepts = physh_obj.get("concepts")
        disciplines = physh_obj.get("disciplines")
        return bool(concepts) or bool(disciplines)
    return False


def summarize_metadata(metadata_path: Path) -> list[dict[str, int]]:
    yearly = defaultdict(lambda: {"total_papers": 0, "physh_papers": 0})
    for record in iter_jsonl(metadata_path):
        if not is_research_article(record):
            continue
        year = paper_year(record)
        if year is None:
            continue
        yearly[year]["total_papers"] += 1
        if has_physh(record):
            yearly[year]["physh_papers"] += 1
    return [{"year": year, **yearly[year]} for year in sorted(yearly)]


def load_rows(path: Path) -> list[dict[str, int]]:
    if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
        return [
            {
                "year": int(row["year"]),
                "total_papers": int(row["total_papers"]),
                "physh_papers": int(row["physh_papers"]),
            }
            for row in iter_jsonl(path)
        ]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        {
            "year": int(row["year"]),
            "total_papers": int(row["total_papers"]),
            "physh_papers": int(row["physh_papers"]),
        }
        for row in payload
    ]


def save_rows(rows: list[dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_plot(
    rows: list[dict[str, int]],
    *,
    output_png: Path | None,
    output_svg: Path | None,
    y_max: int,
    dpi: int,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 5.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )

    valid_rows = [row for row in rows if row["total_papers"] > 0]
    years = np.array([row["year"] for row in valid_rows], dtype=float)
    total = np.array([row["total_papers"] for row in valid_rows], dtype=float)
    physh = np.array([row["physh_papers"] for row in valid_rows], dtype=float)

    pacs_visual = np.zeros_like(total)
    pacs_mask = (years >= PACS_START_YEAR) & (years <= PACS_END_YEAR)
    pacs_visual[pacs_mask] = total[pacs_mask] - physh[pacs_mask]
    neither = total - physh - pacs_visual

    if y_max <= YMIN:
        raise ValueError("y_max must be larger than YMIN.")

    base_w, base_h = DEFAULT_FIGSIZE
    left, right, bottom0, top0 = 0.17, 0.99, 0.28, 0.98
    bottom_margin_in = bottom0 * base_h
    top_margin_in = (1 - top0) * base_h
    axes_h_in = (top0 - bottom0) * base_h
    log_frac = np.log(y_max / YMIN) / np.log(BASE_YMAX / YMIN)
    new_axes_h_in = axes_h_in * log_frac
    new_h = bottom_margin_in + new_axes_h_in + top_margin_in
    bottom = bottom_margin_in / new_h
    top = (bottom_margin_in + new_axes_h_in) / new_h

    fig, ax = plt.subplots(figsize=(base_w, new_h), dpi=dpi)

    def draw_stack(values, bottoms, color, hatch=None, edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW, z=2):
        tops = bottoms + values
        vis_bot = np.maximum(bottoms, YMIN)
        vis_h = tops - vis_bot
        mask = (values > 0) & (tops > YMIN) & (vis_h > 0)
        ax.bar(
            years[mask],
            vis_h[mask],
            bottom=vis_bot[mask],
            width=BAR_WIDTH,
            color=color,
            edgecolor=edgecolor,
            linewidth=lw,
            hatch=hatch,
            zorder=z,
        )

    zeros = np.zeros_like(years)
    draw_stack(neither, zeros, "tab:blue")
    draw_stack(pacs_visual, neither, "tab:orange")
    draw_stack(physh, neither + pacs_visual, "tab:green")

    mask_transition = (years >= PACS_START_YEAR) & (years <= PACS_TRANSITION_END_YEAR)
    draw_stack(
        np.where(mask_transition, pacs_visual, 0),
        neither,
        "none",
        hatch="////",
        edgecolor="0.25",
        lw=0.25,
        z=4,
    )

    ax.set_yscale("log")
    ax.set_yticks([10, 100, 1000, 10000])
    ax.set_yticklabels(["$10^1$", "$10^2$", "$10^3$", "$10^4$"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(YMIN, y_max)
    ax.set_xlim(years.min() - 0.5, 2025.5)
    ax.set_xticks(np.arange(1900, 2026, 25))
    ax.set_ylabel("No. of papers", fontsize=6.2)
    ax.set_xlabel("Year", fontsize=6.2)

    legend_handles = [
        Patch(facecolor="tab:blue", label="No subject headings"),
        Patch(facecolor="tab:orange", hatch="////", edgecolor="0.25", label="PACS transition"),
        Patch(facecolor="tab:orange", label="PACS"),
        Patch(facecolor="tab:green", label="PhySH"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=1, fontsize=5.0, frameon=False)

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    if output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=dpi)
    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot APS subject-heading coverage by year.")
    parser.add_argument("--metadata", help="APS metadata JSONL/JSONL.GZ file.")
    parser.add_argument("--rows", help="Precomputed yearly rows JSON/JSONL with year, total_papers, physh_papers.")
    parser.add_argument("--save-rows", help="Optional JSONL path to save yearly rows.")
    parser.add_argument("--output-png", default="outputs/fig_label_coverage_metadata_2025.png")
    parser.add_argument("--output-svg", default="outputs/fig_label_coverage_metadata_2025.svg")
    parser.add_argument("--y-max", type=int, default=30000)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    if not args.metadata and not args.rows:
        raise SystemExit("Provide either --metadata or --rows.")
    rows = load_rows(Path(args.rows)) if args.rows else summarize_metadata(Path(args.metadata))
    if args.save_rows:
        save_rows(rows, Path(args.save_rows))
    make_plot(
        rows,
        output_png=Path(args.output_png) if args.output_png else None,
        output_svg=Path(args.output_svg) if args.output_svg else None,
        y_max=args.y_max,
        dpi=args.dpi,
    )
    print(f"Wrote label coverage figure -> {args.output_png}")


if __name__ == "__main__":
    main()
