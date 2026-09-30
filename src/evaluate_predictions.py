from __future__ import annotations

import argparse
import json
from pathlib import Path

from physh_tools.metrics import (
    labels_from_jsonl_record,
    load_similarity_npz,
    mean_sample_f1,
    mean_semantic_f1,
    read_jsonl,
)

DEFAULT_SIMILARITY = (
    Path(__file__).resolve().parent.parent / "benchmark" / "concept_similarity.npz"
)


def _row_doi(row: dict) -> str:
    doi = row.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip()
    identifiers = row.get("identifiers")
    if isinstance(identifiers, dict):
        doi = identifiers.get("doi")
        if isinstance(doi, str) and doi.strip():
            return doi.strip()
    return ""


def align_predictions_to_gold(gold_rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    gold_dois = [_row_doi(row) for row in gold_rows]
    pred_dois = [_row_doi(row) for row in pred_rows]

    if len(gold_rows) == len(pred_rows) and gold_dois == pred_dois:
        return pred_rows

    if not all(gold_dois) or not all(pred_dois):
        raise SystemExit(
            f"Gold/prediction length mismatch: {len(gold_rows)} vs {len(pred_rows)}. "
            "Use files in the same order, or include DOI fields so records can be aligned."
        )

    pred_by_doi = {}
    for row, doi in zip(pred_rows, pred_dois):
        pred_by_doi.setdefault(doi, row)

    missing = [doi for doi in gold_dois if doi not in pred_by_doi]
    if missing:
        examples = ", ".join(missing[:5])
        raise SystemExit(
            f"Predictions are missing {len(missing)} gold DOI(s), for example: {examples}"
        )

    return [pred_by_doi[doi] for doi in gold_dois]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySH predictions against gold labels."
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL records.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL records.")
    parser.add_argument(
        "--concept-similarity",
        default=str(DEFAULT_SIMILARITY),
        help="Concept similarity .npz to use for sample Semantic F1.",
    )
    parser.add_argument("--output", default="outputs/evaluation_metrics.json")
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.predictions)
    pred_rows = align_predictions_to_gold(gold_rows, pred_rows)

    gold_disc = [labels_from_jsonl_record(row, "disciplines") for row in gold_rows]
    pred_disc = [labels_from_jsonl_record(row, "disciplines") for row in pred_rows]
    gold_conc = [labels_from_jsonl_record(row, "concepts") for row in gold_rows]
    pred_conc = [labels_from_jsonl_record(row, "concepts") for row in pred_rows]

    metrics = {
        "num_items": len(gold_rows),
        "discipline_sample_f1": mean_sample_f1(pred_disc, gold_disc),
        "concept_exact_sample_f1": mean_sample_f1(pred_conc, gold_conc),
    }

    if args.concept_similarity:
        labels, similarity = load_similarity_npz(args.concept_similarity)
        metrics.update(
            {
                f"concept_{key}": value
                for key, value in mean_semantic_f1(
                    pred_conc,
                    gold_conc,
                    labels=labels,
                    similarity=similarity,
                ).items()
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
