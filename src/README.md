# Script Guide

Entry points in this directory; shared helpers in `physh_tools/`.

## Classification

- `classify_one_paper.py`: classify one title/abstract pair into PhySH disciplines and concepts.
- `classify_batch.py`: classify a JSONL file of papers with the same prompt and label constraints.

Both accept `--model`, `--temperature`, `--top-p`, `--top-k`, and optional `--examples` for few-shot labeled examples. Pass the same benchmark file to compare models.

## Embedding and UMAP

- `embed_one_paper.py`: embed one title/abstract pair with `gemini-embedding-2`.
- `embed_papers.py`: embed a JSONL corpus and write `embeddings.npy`, `manifest.csv`, and `meta.json`.
- `reduce_umap.py`: reduce a corpus embedding matrix to two UMAP coordinates per paper.

Plotting scripts are in `../plotting/`.

## Evaluation

- `evaluate_predictions.py`: compute exact discipline F1 and concept Semantic F1 using `benchmark/concept_similarity.npz`.

Run all scripts from the repository root:

```bash
python src/classify_one_paper.py --title "..." --abstract "..."
```
