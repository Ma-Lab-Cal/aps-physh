from __future__ import annotations

import argparse

from physh_tools.classification import DEFAULT_MODEL, classify_items
from physh_tools.io import load_env_file, paper_doi, paper_id, paper_text, paper_year, read_jsonl, write_jsonl
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
    parser = argparse.ArgumentParser(description="Classify a JSONL file of papers into PhySH labels.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples", help="Optional JSONL file of few-shot examples with gold PhySH labels.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)
    exemplars = load_exemplars(args.examples)
    records = read_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    items = []
    for record in records:
        text = paper_text(record)
        if not text:
            continue
        items.append({"paper_id": paper_id(record, text), "doi": paper_doi(record), "year": paper_year(record), "text": text})

    outputs = []
    for start in range(0, len(items), args.batch_size):
        outputs.extend(
            classify_items(
                items[start : start + args.batch_size],
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                exemplars=exemplars,
            )
        )
        print(f"Classified {min(start + args.batch_size, len(items))}/{len(items)} papers")

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} classifications -> {args.output}")


if __name__ == "__main__":
    main()
