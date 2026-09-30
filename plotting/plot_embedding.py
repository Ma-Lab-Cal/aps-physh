from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from physh_tools.physh import UNKNOWN_COLOR, discipline_colors


def open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return open(path, "r", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UMAP coordinates colored by primary PhySH discipline.")
    parser.add_argument("--coordinates", required=True, help="CSV with paper_id/doi/x/y.")
    parser.add_argument("--labels", required=True, help="JSONL with paper_id/doi/predicted_disciplines.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    coords = pd.read_csv(args.coordinates)
    if "paper_id" in coords.columns:
        coords = coords.drop_duplicates(subset=["paper_id"])
    elif "doi" in coords.columns:
        coords = coords.drop_duplicates(subset=["doi"])
    else:
        raise SystemExit("Coordinates must contain either `doi` or `paper_id`.")

    label_rows = []
    with open_text(args.labels) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            disciplines = row.get("predicted_disciplines") or []
            if disciplines:
                label_rows.append({"paper_id": row.get("paper_id", ""), "doi": row.get("doi", ""), "Discipline": disciplines[0]})
    labels = pd.DataFrame(label_rows)
    if labels.empty:
        raise SystemExit("No labels with predicted_disciplines found.")

    if "paper_id" in coords.columns and "paper_id" in labels.columns:
        merged = coords.merge(labels[["paper_id", "Discipline"]], on="paper_id", how="left")
    else:
        merged = coords.copy()
        merged["Discipline"] = pd.NA

    if "doi" in coords.columns:
        doi_labels = labels[labels["doi"].astype(str) != ""][["doi", "Discipline"]].drop_duplicates("doi")
        if not doi_labels.empty:
            merged["doi"] = merged["doi"].astype(str)
            doi_labels["doi"] = doi_labels["doi"].astype(str)
            doi_lookup = merged.merge(doi_labels, on="doi", how="left", suffixes=("", "_by_doi"))
            merged["Discipline"] = merged["Discipline"].fillna(doi_lookup["Discipline_by_doi"])
    merged = merged.dropna(subset=["Discipline"]).copy()
    if merged.empty:
        raise SystemExit("No overlap between coordinates and labels.")

    colors = discipline_colors()
    order = [d for d in colors if d in set(merged["Discipline"])]
    for discipline in merged["Discipline"].value_counts().index:
        if discipline not in order:
            order.append(discipline)
    palette = [colors.get(discipline, UNKNOWN_COLOR) for discipline in order]

    plot_df = merged.sample(frac=1.0, random_state=42)
    sns.set_style("white")
    plt.figure(figsize=(10, 10), dpi=args.dpi)
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="Discipline",
        hue_order=order,
        palette=palette,
        s=2.5,
        alpha=0.2,
        linewidth=0,
        legend=False,
    )
    x_min, x_max = plot_df["x"].quantile([0.01, 0.99])
    y_min, y_max = plot_df["y"].quantile([0.01, 0.99])
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Plotted {len(plot_df)} points -> {args.output}")


if __name__ == "__main__":
    main()
