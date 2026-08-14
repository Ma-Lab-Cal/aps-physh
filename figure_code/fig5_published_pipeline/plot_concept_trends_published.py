#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the PUBLISHED Figure 5 of LT19584 from local public-repo data.

Local adaptation of the PI's Colab notebook `reports/fig5.py` (the code that
generated the published figure). Pipeline, verbatim from the notebook:

  * five "backfilled" model runs for years <= 2016  -> aps-physh-main/runs_v1_full/run_{1..5}.jsonl.gz
    (the Drive files carry `item_key` + `year` inline; the local release files
    carry `doi` only, so publication years are joined from data/aps_index.sqlite)
  * consolidated-metadata ground truth for 2017+    -> aps_metadata_1893_2025_raw.jsonl
    (author-assigned PhySH concept UUIDs mapped to names via PhySH/id_to_concepts.json,
    the local copy of the notebook's `physh.json`)
  * counting method "average" (mean paper count across the 5 runs), share of papers,
    3-year centered rolling mean.

Stages (cached so plot iterations are fast):
  python plot_concept_trends_published.py --stage aggregate   # build cache/*.pkl.gz
  python plot_concept_trends_published.py --stage plot        # fig5_reproduced.png
  python plot_concept_trends_published.py --stage plot --revised
        # fig5_revised.png: "Mathematical physics" removed from panel (c) ONLY

The published PNG additionally carries panel letters (a)-(d) and was saved with
pad_inches=0; both are replicated here (the notebook export lacks the letters).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

RUNS_DIR = PROJECT / "aps-physh-main" / "runs_v1_full"
DRIVE_RUNS_DIR = Path(os.environ.get("FIG5_DRIVE_RUNS", "/tmp/fig5_drive/backfilled runs"))
INDEX_SQLITE = PROJECT / "data" / "aps_index.sqlite"
RAW_METADATA = PROJECT / "aps_metadata_1893_2025_raw.jsonl"
TAXONOMY_JSON = PROJECT / "PhySH" / "id_to_concepts.json"
CACHE_DIR = HERE / "cache"

GOLD_START_YEAR = 2017
TAG_RE = re.compile(r"<[^>]+>")
YEAR_RE = re.compile(r"(18|19|20)\d{2}")
EXCLUDED_TITLE_PREFIXES = (
    "erratum",
    "correction to",
    "comment on",
    "reply to",
    "publisher's note",
    "editorial",
    "announcement",
    "note.",
    "retraction",
)
EXCLUDED_TITLE_EXACTS = {
    "notes",
    "new books",
    "minor contributions",
    "books received",
    "proceedings",
    "errata",
}


# ----------------------------------------------------------------------------
# Notebook helpers (verbatim from reports/fig5.py)
# ----------------------------------------------------------------------------

def clean_text(value):
    if not isinstance(value, str):
        return ""
    text = TAG_RE.sub(" ", value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def normalize_title(title):
    text = (title or "").strip().lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text)


def is_excluded_title(title):
    text = normalize_title(title)
    if not text:
        return False
    if text in EXCLUDED_TITLE_EXACTS:
        return True
    return any(text.startswith(prefix) for prefix in EXCLUDED_TITLE_PREFIXES)


def extract_title(obj):
    title = obj.get("title")
    if isinstance(title, dict):
        return clean_text(title.get("value"))
    return clean_text(title)


def extract_year(obj):
    for key in ("year", "date", "published", "pub_date", "publicationDate"):
        value = obj.get(key)
        if isinstance(value, int) and 1800 <= value <= 2100:
            return value
        if isinstance(value, str):
            match = YEAR_RE.search(value)
            if match:
                return int(match.group(0))
    return None


def extract_journal_id(obj):
    journal = obj.get("journal")
    if isinstance(journal, dict):
        value = journal.get("id")
        if isinstance(value, str):
            return value.strip()
    return ""


def extract_physh_concepts(obj, taxonomy):
    physh = ((obj.get("classificationSchemes") or {}).get("physh")) or {}
    concepts = []
    for entry in physh.get("concepts") or []:
        label = ""
        if isinstance(entry, dict):
            label = clean_text(entry.get("label"))
            if not label and "id" in entry:
                concept_id = entry["id"]
                if concept_id in taxonomy:
                    label = clean_text(taxonomy[concept_id].get("name", concept_id))
                else:
                    label = concept_id
        else:
            label = clean_text(entry)
        if label:
            concepts.append(label)
    return list(dict.fromkeys(concepts))


# ----------------------------------------------------------------------------
# Stage 1: aggregation (with caches)
# ----------------------------------------------------------------------------

def load_doi_years() -> dict:
    con = sqlite3.connect(str(INDEX_SQLITE))
    years = dict(con.execute("SELECT doi, year FROM papers"))
    con.close()
    return years


def aggregate_runs(source: str = "public") -> None:
    """Notebook cell 3: process backfilled runs (<= 2016).

    source="public": released aps-physh-main/runs_v1_full/run_{1..5}.jsonl.gz.
      These lack an inline `year` (joined from the APS index by DOI) and carry
      canonical-filtered concept labels.
    source="drive": the notebook's exact inputs (backfilled_run_*.jsonl with
      `item_key` + `year` inline, run_1 replaced by the newer file), for
      verification against the published PNG.
    """
    if source == "drive":
        run_paths = sorted(DRIVE_RUNS_DIR.glob("backfilled_run_*.jsonl"))
        if len(run_paths) != 5:
            raise SystemExit(f"expected 5 run files in {DRIVE_RUNS_DIR}, found {len(run_paths)}")
        doi_years = None
    else:
        run_paths = sorted(RUNS_DIR.glob("run_*.jsonl.gz"))
        if len(run_paths) != 5:
            raise SystemExit(f"expected 5 run files in {RUNS_DIR}, found {len(run_paths)}")
        doi_years = load_doi_years()

    paper_concept_counts = defaultdict(lambda: defaultdict(int))
    paper_years = {}
    for filepath in run_paths:
        print(f"Processing {filepath.name}...")
        opener = (lambda p: gzip.open(p, "rt", encoding="utf-8")) if filepath.suffix == ".gz" \
            else (lambda p: open(p, "r", encoding="utf-8"))
        with opener(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if doi_years is None:
                    item_key = str(row["item_key"])
                    year = int(row["year"])
                else:
                    item_key = str(row["doi"])
                    year = doi_years.get(item_key)
                    if year is None:
                        continue
                    year = int(year)
                if year >= GOLD_START_YEAR:
                    continue
                paper_years[item_key] = year
                for concept in row.get("predicted_concepts", []):
                    paper_concept_counts[item_key][concept] += 1

    pred_concept_year_hist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pred_year_totals = defaultdict(int)
    for item_key, year in paper_years.items():
        pred_year_totals[year] += 1
        for concept, count in paper_concept_counts[item_key].items():
            pred_concept_year_hist[concept][year][count] += 1

    payload = {
        "pred_concept_year_hist": {
            c: {y: dict(h) for y, h in ymap.items()}
            for c, ymap in pred_concept_year_hist.items()
        },
        "pred_year_totals": dict(pred_year_totals),
        "num_runs": len(run_paths),
        "n_papers": len(paper_years),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_DIR / pred_cache_name(source), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Predicted papers: {len(paper_years)}; years "
          f"{min(pred_year_totals)}-{max(pred_year_totals)}")


def pred_cache_name(source: str) -> str:
    return "pred_agg.pkl.gz" if source == "public" else f"pred_agg_{source}.pkl.gz"


def aggregate_gold() -> None:
    """Notebook cell 4: ground truth 2017+ from consolidated APS metadata."""
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        physh_taxonomy = json.load(f)

    gold_concept_year_counts = defaultdict(lambda: defaultdict(int))
    gold_year_totals = defaultdict(int)
    n_lines = 0
    print(f"Processing ground truth from {RAW_METADATA} ...")
    with open(RAW_METADATA, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            # Only records carrying a physh block can contribute (papers with no
            # mapped concepts are skipped by the notebook, including from totals).
            if '"physh"' not in line:
                continue
            obj = json.loads(line)
            year = extract_year(obj)
            if year is None or year < GOLD_START_YEAR:
                continue
            if extract_journal_id(obj) == "PHYSICS":
                continue
            if is_excluded_title(extract_title(obj)):
                continue
            concepts = extract_physh_concepts(obj, physh_taxonomy)
            if not concepts:
                continue
            gold_year_totals[year] += 1
            for c in concepts:
                gold_concept_year_counts[c][year] += 1

    payload = {
        "gold_concept_year_counts": {c: dict(y) for c, y in gold_concept_year_counts.items()},
        "gold_year_totals": dict(gold_year_totals),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_DIR / "gold_agg.pkl.gz", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Scanned {n_lines} metadata lines; gold papers 2017+: "
          f"{sum(gold_year_totals.values())}; years "
          f"{min(gold_year_totals)}-{max(gold_year_totals)}")


# ----------------------------------------------------------------------------
# Stage 2: plotting (verbatim notebook logic)
# ----------------------------------------------------------------------------

def load_caches(source: str = "public"):
    with gzip.open(CACHE_DIR / pred_cache_name(source), "rb") as f:
        pred = pickle.load(f)
    with gzip.open(CACHE_DIR / "gold_agg.pkl.gz", "rb") as f:
        gold = pickle.load(f)
    return pred, gold


def make_figure(revised: bool, output: Path, letters: dict | None,
                source: str = "public") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    pred, gold = load_caches(source)
    pred_concept_year_hist = pred["pred_concept_year_hist"]
    pred_year_totals = pred["pred_year_totals"]
    num_runs = pred["num_runs"]
    gold_concept_year_counts = gold["gold_concept_year_counts"]
    gold_year_totals = gold["gold_year_totals"]

    all_years = sorted(set(pred_year_totals) | set(gold_year_totals))
    all_concepts = set(pred_concept_year_hist) | set(gold_concept_year_counts)

    def predicted_value_for_year(concept, year, counting_method="consensus_3"):
        hist = pred_concept_year_hist.get(concept, {}).get(year, {})
        if counting_method == "average":
            return sum(k * v for k, v in hist.items()) / num_runs
        if counting_method == "union":
            return sum(v for k, v in hist.items() if k >= 1)
        if counting_method == "consensus_3":
            return sum(v for k, v in hist.items() if k >= 3)
        if counting_method == "consensus_5":
            return sum(v for k, v in hist.items() if k == 5)
        raise ValueError(f"Unknown counting_method: {counting_method}")

    def combined_value_for_year(concept, year, counting_method="consensus_3"):
        if year >= GOLD_START_YEAR:
            return gold_concept_year_counts.get(concept, {}).get(year, 0)
        return predicted_value_for_year(concept, year, counting_method=counting_method)

    def total_papers_for_year(year):
        if year >= GOLD_START_YEAR:
            return gold_year_totals.get(year, 0)
        return pred_year_totals.get(year, 0)

    def get_shares(concepts, method="consensus_3"):
        shares_dict = {}
        for c in concepts:
            if c in all_concepts:
                vals = [combined_value_for_year(c, y, method) for y in all_years]
                shares = [(v / total_papers_for_year(y) * 100 if total_papers_for_year(y) else 0)
                          for v, y in zip(vals, all_years)]
                shares_dict[c] = pd.Series(shares).rolling(window=3, center=True, min_periods=1).mean()
        return shares_dict

    # --- Nobel Prize Concepts Data ---
    nobel_concepts = [
        "X-ray diffraction",
        "Deep inelastic scattering",
        "Cosmic microwave background",
        "Graphene",
    ]
    discovery_yrs = {
        "X-ray diffraction": 1912,
        "Deep inelastic scattering": 1968,
        "Graphene": 2004,
        "Cosmic microwave background": 1964,
    }
    nobel_yrs = {
        "X-ray diffraction": 1914,
        "Deep inelastic scattering": 1990,
        "Graphene": 2010,
        "Cosmic microwave background": 1978,
    }

    # --- Modern Fields Data ---
    new_fields = [
        "Topological insulators",
        "Quantum information processing",
        "Machine learning",
        "Gravitational wave sources",
        "Spintronics",
    ]

    FIG_W = 7.1
    LINE_W = 1.0
    CURVE_ALPHA = 0.80
    EVENT_LW = 0.75
    EVENT_ALPHA = 0.40
    COL_SPACE = 0.20
    LEFT_MARGIN = 0.085
    RIGHT_MARGIN = 0.985
    BOTTOM_MARGIN = 0.085
    TOP_MARGIN = 0.975

    plt.style.use("default")

    def short_label(label):
        replacements = {
            "Quantum chromodynamics": "QCD",
            "Reaction models & methods for nuclear reactions": "Nuclear reactions",
            "Density functional theory": "DFT",
            "Monte Carlo methods": "Monte Carlo",
            "High-temperature superconductors": r"High-$T_c$ superconductors",
            "Green's function methods": "Green's function",
            "Tensor network methods": "Tensor networks",
            "Perturbation theory": "Perturbation\ntheory",
            "Mathematical physics methods": "Mathematical physics",
        }
        return replacements.get(label, label)

    def add_inner_title(ax, title, inner_title_fs, show_title_frame=True,
                        title_frame_alpha=0.72, title_frame_edge_alpha=0.40,
                        title_frame_lw=0.35):
        if show_title_frame:
            title_bbox = dict(
                boxstyle="round,pad=0.18,rounding_size=0.03",
                facecolor="white",
                edgecolor=(0.55, 0.55, 0.55, title_frame_edge_alpha),
                linewidth=title_frame_lw,
                alpha=title_frame_alpha,
            )
        else:
            title_bbox = None
        ax.text(0.035, 0.955, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=inner_title_fs, fontweight="normal", bbox=title_bbox, zorder=12)

    def add_inner_legend(ax, legend_fs, legend_x, legend_y, legend_loc="upper left",
                         show_legend_frame=True, legend_frame_alpha=0.72,
                         legend_frame_edge_alpha=0.40, legend_frame_lw=0.35):
        leg = ax.legend(
            loc=legend_loc,
            bbox_to_anchor=(legend_x, legend_y),
            handlelength=1.8,
            borderaxespad=0.0,
            labelspacing=0.25,
            handletextpad=0.45,
            borderpad=0.35 if show_legend_frame else 0.0,
            frameon=show_legend_frame,
            fancybox=True,
            fontsize=legend_fs,
        )
        leg.set_zorder(12)
        if show_legend_frame:
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_alpha(legend_frame_alpha)
            frame.set_edgecolor((0.55, 0.55, 0.55, legend_frame_edge_alpha))
            frame.set_linewidth(legend_frame_lw)
        return leg

    def get_data_max(data_arrays):
        vals = np.concatenate([np.asarray(v, dtype=float) for v in data_arrays])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 1
        ymax = vals.max()
        if ymax <= 0:
            ymax = 1
        return ymax

    def set_custom_y_ticks(ax, data_arrays, y_tick_top, y_tick_step, y_tick_top_pos,
                           y_bottom_pad_frac=0.04, expand_ticks_to_data=False,
                           tick_format="%g"):
        if y_tick_step <= 0:
            raise ValueError("y_tick_step must be positive.")
        if not (0 < y_tick_top_pos < 1):
            raise ValueError("y_tick_top_pos must be between 0 and 1.")
        data_max = get_data_max(data_arrays)
        if expand_ticks_to_data and data_max > y_tick_top:
            y_tick_top = np.ceil(data_max / y_tick_step) * y_tick_step
        ticks = np.arange(0, y_tick_top + 0.5 * y_tick_step, y_tick_step)
        y_min = -abs(y_tick_top) * y_bottom_pad_frac
        y_max = y_min + (y_tick_top - y_min) / y_tick_top_pos
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))

    def polish_axis(ax, axis_label_fs):
        ax.grid(False)
        ax.set_xlabel("Year", fontsize=axis_label_fs)
        ax.set_ylabel("Share of papers (%)", fontsize=axis_label_fs)
        ax.tick_params(direction="out", top=False, right=False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    def color_map_from_list(labels, colors):
        return {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    # ------------------------------------------------------------------
    # Data preparation (notebook, `method="average"`)
    # ------------------------------------------------------------------
    target_nuclear = (
        "Nuclear reactions"
        if "Nuclear reactions" in all_concepts
        else "Reaction models & methods for nuclear reactions"
    )
    group_1_concepts = ["Atomic spectra", target_nuclear, "Mesons", "Quantum chromodynamics"]

    traditional_theory = [
        "Mathematical physics methods",
        "Perturbation theory",
        "Green's function methods",
    ]
    if revised:
        # Revision: "Mathematical physics methods" replaced by "Renormalization group"
        # in panel (c) ONLY (per-concept precision 0.06 vs 0.41; SM per-concept table).
        traditional_theory = [
            "Renormalization group",
            "Perturbation theory",
            "Green's function methods",
        ]

    comp_methods = ["Density functional theory", "Monte Carlo methods", "Tensor network methods"]

    s1_data = get_shares(group_1_concepts, method="average")
    trad_data = get_shares(traditional_theory, method="average")
    comp_data_subset = get_shares(comp_methods, method="average")
    n_data = get_shares(nobel_concepts, method="average")
    e_data = get_shares(new_fields, method="average")

    years = np.asarray(all_years)

    if revised:
        # Byte-parallel to the original label ("Renormalization group" is one
        # character longer than "Mathematical physics"; same wrap behavior).
        analytical_label = (
            "Renormalization group + "
            + short_label(traditional_theory[1])
            + " + "
            + short_label(traditional_theory[2])
        )
    else:
        analytical_label = (
            "Mathematical physics + "
            + short_label(traditional_theory[1])
            + " + "
            + short_label(traditional_theory[2])
        )
    computational_label = " + ".join([short_label(c) for c in comp_methods])

    tab10 = plt.get_cmap("tab10").colors
    palette = {
        "colors": [tab10[0], tab10[1], tab10[2], tab10[3], tab10[7]],
        "panel_c": [tab10[0], tab10[1]],
    }

    # ------------------------------------------------------------------
    # make_concept_plot (notebook, with the final call's parameter values)
    # ------------------------------------------------------------------
    base_fs = 8
    inner_title_fs = 7
    axis_label_fs = 7
    tick_fs = 7
    legend_a_fs = legend_b_fs = legend_c_fs = legend_d_fs = 6
    row_space = 0.08
    panel_height = 1.9
    y_bottom_pad_frac = 0.04

    fig_h = 2 * panel_height

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base_fs,
        "axes.titlesize": inner_title_fs,
        "axes.labelsize": axis_label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.dpi": 500,
        "savefig.dpi": 900,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axs = plt.subplots(2, 2, figsize=(FIG_W, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, bottom=BOTTOM_MARGIN,
                        top=TOP_MARGIN, wspace=COL_SPACE, hspace=row_space)

    base_colors = palette["colors"]
    colors_a = color_map_from_list(list(s1_data.keys()), base_colors)
    colors_b = color_map_from_list(list(n_data.keys()), base_colors)
    colors_d = color_map_from_list(list(e_data.keys()), base_colors)
    if "Spintronics" in colors_d and "Gravitational wave sources" in colors_d:
        c_spin = colors_d["Spintronics"]
        c_grav = colors_d["Gravitational wave sources"]
        colors_d["Spintronics"] = c_grav
        colors_d["Gravitational wave sources"] = c_spin

    # Panel A
    ax = axs[0, 0]
    for c, vals in s1_data.items():
        ax.plot(years, vals, label=short_label(c), color=colors_a[c], lw=LINE_W, alpha=CURVE_ALPHA)
    ax.set_xlim(years.min(), years.max())
    set_custom_y_ticks(ax, data_arrays=s1_data.values(), y_tick_top=16, y_tick_step=4,
                       y_tick_top_pos=0.9, y_bottom_pad_frac=y_bottom_pad_frac,
                       expand_ticks_to_data=False, tick_format="%g")
    add_inner_title(ax, "Atomic, Nuclear,\n& Particle Research", inner_title_fs,
                    show_title_frame=False)
    add_inner_legend(ax, legend_a_fs, 0.975, 0.955, legend_loc="upper right",
                     show_legend_frame=False)

    # Panel B
    ax = axs[0, 1]
    for c, vals in n_data.items():
        color = colors_b[c]
        ax.plot(years, vals, label=short_label(c), color=color, lw=LINE_W, alpha=CURVE_ALPHA)
        ax.axvline(discovery_yrs[c], color=color, alpha=EVENT_ALPHA, linestyle="-",
                   lw=EVENT_LW, zorder=0)
        ax.axvline(nobel_yrs[c], color=color, alpha=EVENT_ALPHA, linestyle=(0, (4, 2)),
                   lw=EVENT_LW, zorder=0)
    ax.set_xlim(years.min(), years.max())
    set_custom_y_ticks(ax, data_arrays=n_data.values(), y_tick_top=5, y_tick_step=1,
                       y_tick_top_pos=0.87, y_bottom_pad_frac=y_bottom_pad_frac,
                       expand_ticks_to_data=False, tick_format="%g")
    add_inner_title(ax, "Nobel Prize:\nDiscovery vs. Award", inner_title_fs,
                    show_title_frame=True, title_frame_edge_alpha=0.0, title_frame_lw=0.0)
    add_inner_legend(ax, legend_b_fs, 0.975, 0.955, legend_loc="upper right",
                     show_legend_frame=True, legend_frame_edge_alpha=0.0, legend_frame_lw=0.0)

    # Panel C
    ax = axs[1, 0]
    sum_trad = np.sum(np.vstack(list(trad_data.values())), axis=0)
    sum_comp = np.sum(np.vstack(list(comp_data_subset.values())), axis=0)
    ax.plot(years, sum_trad, label=analytical_label, color=palette["panel_c"][0],
            lw=LINE_W, alpha=CURVE_ALPHA)
    ax.plot(years, sum_comp, label=computational_label, color=palette["panel_c"][1],
            lw=LINE_W, alpha=CURVE_ALPHA)
    ax.set_xlim(years.min(), 2025)
    set_custom_y_ticks(ax, data_arrays=[sum_trad, sum_comp], y_tick_top=8, y_tick_step=2,
                       y_tick_top_pos=0.999999, y_bottom_pad_frac=y_bottom_pad_frac,
                       expand_ticks_to_data=False, tick_format="%g")
    add_inner_title(ax, "Analytical vs. Computational", inner_title_fs, show_title_frame=False)
    add_inner_legend(ax, legend_c_fs, 0.035, 0.855, legend_loc="upper left",
                     show_legend_frame=False)

    # Panel D
    ax = axs[1, 1]
    for c, vals in e_data.items():
        ax.plot(years, vals, label=short_label(c), color=colors_d[c], lw=LINE_W,
                alpha=CURVE_ALPHA)
    ax.set_xlim(years.min(), years.max())
    set_custom_y_ticks(ax, data_arrays=e_data.values(), y_tick_top=2, y_tick_step=1,
                       y_tick_top_pos=0.9, y_bottom_pad_frac=y_bottom_pad_frac,
                       expand_ticks_to_data=False, tick_format="%g")
    add_inner_title(ax, "Surge of Modern Era Concepts", inner_title_fs, show_title_frame=False)
    add_inner_legend(ax, legend_d_fs, 0.035, 0.855, legend_loc="upper left",
                     show_legend_frame=False)

    # Axis styling
    for i, ax in enumerate(axs.flat):
        polish_axis(ax, axis_label_fs)
        if i in [0, 1]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

    # Panel letters (present in the published PNG; not in the notebook export).
    # The published letters are Arial-metric (Arimo) ~10pt, individually placed;
    # anchor fractions below reproduce the published ink positions exactly.
    if letters:
        fs = letters_fontsize(letters)
        family = letters_family(letters)
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(ARIMO_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not register Arimo ({exc}); letters use default font")
            family = plt.rcParams["font.family"]
        for text, pos in letters.items():
            if text.startswith("__"):
                continue
            x, y = pos
            fig.text(x, y, text, fontsize=fs, family=family, ha="left", va="top")

    output.parent.mkdir(parents=True, exist_ok=True)
    # The published PNG is not a matplotlib bbox_inches="tight" export: it is the
    # full 6390x3420 canvas (7.1x3.8 in @ 900 dpi) cropped to the content with a
    # few px of margin (letters were added and the image cropped outside
    # matplotlib). Render the full canvas and apply that exact crop.
    fig.savefig(output, dpi=900)
    plt.close(fig)
    from PIL import Image
    im = Image.open(output)
    if im.size == (6390, 3420):
        im.crop(PUBLISHED_CROP).save(output)
    else:
        print(f"warning: canvas {im.size} != (6390, 3420); skipping crop")
    print(f"wrote {output}")


ARIMO_PATH = "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"

# Canvas-pixel crop replicating the published framing: content at offset
# (L=189, T=34), published size 6115x3380.
PUBLISHED_CROP = (189, 34, 189 + 6115, 34 + 3380)


def letters_fontsize(letters):
    return letters.get("__fontsize__", 10)


def letters_family(letters):
    return letters.get("__family__", "Arimo")


# Anchor fractions on the 6390x3420 canvas -> published letter ink positions
# ((a) ink at canvas (196,40) etc.; Arimo fs10 ha=left va=top ink offset dx=+9,dy=0).
DEFAULT_LETTERS = {
    "(a)": (187 / 6390, 1 - 40 / 3420),
    "(b)": (3363 / 6390, 1 - 40 / 3420),
    "(c)": (187 / 6390, 1 - 1603 / 3420),
    "(d)": (3371 / 6390, 1 - 1608 / 3420),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["aggregate", "plot", "all"], default="plot")
    ap.add_argument("--data", choices=["public", "drive"], default="public",
                    help="pre-2017 run source: released repo files or the notebook's Drive files")
    ap.add_argument("--revised", action="store_true",
                    help="remove Mathematical physics from panel (c)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--letters-json", default=None,
                    help="JSON file with {'(a)': [x,y], ..., '__fontsize__': fs}")
    ap.add_argument("--no-letters", action="store_true")
    args = ap.parse_args()

    if args.stage in ("aggregate", "all"):
        if not (CACHE_DIR / pred_cache_name(args.data)).exists():
            aggregate_runs(args.data)
        else:
            print("pred cache exists, skipping (delete to rebuild)")
        if not (CACHE_DIR / "gold_agg.pkl.gz").exists():
            aggregate_gold()
        else:
            print("gold cache exists, skipping (delete to rebuild)")

    if args.stage in ("plot", "all"):
        letters = None
        if not args.no_letters:
            letters = dict(DEFAULT_LETTERS)
            if args.letters_json:
                with open(args.letters_json) as f:
                    loaded = json.load(f)
                letters = {k: (tuple(v) if isinstance(v, list) else v) for k, v in loaded.items()}
        # Default name carries the data source so reruns cannot silently
        # overwrite a deliverable rendered from the other variant; the
        # canonical deliverable names are passed explicitly (see RECIPE.md).
        default_name = f"fig5_{args.data}{'_revised' if args.revised else ''}.png"
        out = args.output or str(HERE / default_name)
        make_figure(args.revised, Path(out), letters, source=args.data)


if __name__ == "__main__":
    main()
