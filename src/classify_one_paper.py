from __future__ import annotations

import argparse
import json
from pathlib import Path

from physh_tools.classification import DEFAULT_MODEL, classify_items
from physh_tools.io import load_env_file, paper_doi, paper_id, paper_text, paper_year, read_jsonl
from physh_tools.metrics import labels_from_jsonl_record


def load_exemplars(path: str | None) -> list[dict[str, object]]:
    if not path:
        return []
    exemplars = []
    for record in read_jsonl(path):
        text = paper_text(record)
        disciplines = labels_from_jsonl_record(record, "disciplines")
        concepts = labels_from_jsonl_record(record, "concepts")
        if text and disciplines and concepts:
            exemplars.append(
                {
                    "text_input": text,
                    "disciplines": disciplines,
                    "concepts": concepts,
                }
            )
    return exemplars


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one paper into PhySH disciplines and concepts.")
    parser.add_argument("--title", default="")
    parser.add_argument("--abstract", default="")
    parser.add_argument("--doi", default="")
    parser.add_argument("--year", default="")
    parser.add_argument("--input-json", help="Optional JSON file containing one paper record.")
    parser.add_argument("--output", default="outputs/one_paper_classification.json")
    parser.add_argument("--examples", help="Optional JSONL file of few-shot examples with gold PhySH labels.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if args.input_json:
        record = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        record = {"title": args.title, "abstract": args.abstract, "doi": args.doi, "year": args.year}

    text = paper_text(record)
    if not text:
        raise SystemExit("Provide --title and/or --abstract, or --input-json with usable text.")
    item = {
        "paper_id": paper_id(record, text),
        "doi": paper_doi(record),
        "year": paper_year(record),
        "text": text,
    }
    result = classify_items(
        [item],
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        exemplars=load_exemplars(args.examples),
    )[0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
