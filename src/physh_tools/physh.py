from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = PACKAGE_ROOT / "taxonomy"
STYLE_DIR = PACKAGE_ROOT / "style"

UNKNOWN_COLOR = "#d3d3d3"


def load_taxonomy_file(name: str) -> dict:
    with (TAXONOMY_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discipline_labels() -> list[str]:
    return sorted(set(load_taxonomy_file("disciplines.json").values()))


def concept_labels() -> list[str]:
    concepts = load_taxonomy_file("concepts.json")
    return sorted({str(payload["label"]) for payload in concepts.values()})


def facet_labels() -> list[str]:
    return sorted(set(load_taxonomy_file("facets.json").values()))


def discipline_colors() -> dict[str, str]:
    with (STYLE_DIR / "discipline_colors.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def color_for_discipline(label: str) -> str:
    return discipline_colors().get(label, UNKNOWN_COLOR)
