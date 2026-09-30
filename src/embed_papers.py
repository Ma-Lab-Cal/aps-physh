from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from physh_tools.embedding import embed_texts
from physh_tools.io import load_env_file, paper_doi, paper_id, paper_text, paper_year, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed user-provided papers with gemini-embedding-2.")
    parser.add_argument("--input", required=True, help="JSONL file with title/abstract records.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-embedding-2")
    parser.add_argument("--output-dimensionality", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--no-normalize", action="store_true", help="Disable row-wise L2 normalization.")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)
    records = read_jsonl(args.input)
    rows = []
    texts = []
    for idx, record in enumerate(records):
        text = paper_text(record)
        if not text:
            continue
        row = {
            "row_idx": idx,
            "paper_id": paper_id(record, text),
            "doi": paper_doi(record),
            "year": paper_year(record),
            "text_input": text,
        }
        rows.append(row)
        texts.append(text)

    if not rows:
        raise SystemExit("No papers with usable title/abstract text were found.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = embed_texts(
        texts,
        model=args.model,
        output_dimensionality=args.output_dimensionality,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )
    np.save(out_dir / "embeddings.npy", vectors)

    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_idx", "paper_id", "doi", "year", "text_input"])
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "output_dimensionality": args.output_dimensionality,
                "normalized": not args.no_normalize,
                "count": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Embedded {len(rows)} papers -> {out_dir}")


if __name__ == "__main__":
    main()
