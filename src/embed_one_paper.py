from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physh_tools.embedding import embed_texts
from physh_tools.io import load_env_file, paper_doi, paper_id, paper_text, paper_year


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed one paper with gemini-embedding-2.")
    parser.add_argument("--title", default="")
    parser.add_argument("--abstract", default="")
    parser.add_argument("--doi", default="")
    parser.add_argument("--year", default="")
    parser.add_argument("--input-json", help="Optional JSON file containing one paper record.")
    parser.add_argument("--output-dir", default="outputs/one_paper_embedding")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if args.input_json:
        record = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        record = {
            "title": args.title,
            "abstract": args.abstract,
            "doi": args.doi,
            "year": args.year,
        }

    text = paper_text(record)
    if not text:
        raise SystemExit("Provide --title and/or --abstract, or --input-json with usable text.")

    vector = embed_texts([text])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embedding.npy", vector)

    metadata = {
        "paper_id": paper_id(record, text),
        "doi": paper_doi(record),
        "year": paper_year(record),
        "text_input": text,
        "model": "gemini-embedding-2",
        "output_dimensionality": int(vector.shape[1]),
        "normalized": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote one embedding with shape {vector.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
