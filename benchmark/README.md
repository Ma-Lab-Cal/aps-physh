# Benchmark Artifacts

Frozen artifacts for reproducing the paper's F1 scores.

## Files

- `concept_similarity.npz`: read by `src/evaluate_predictions.py`. Contains two arrays: `labels` (length 3,793) and `similarity` (3,793 × 3,793, `float32`, values in `[0, 1]`, diagonal 1).
- `ground_truth.jsonl.gz`: 197,167 post-2016 APS research articles with APS-assigned PhySH labels. Each row contains `doi`, `title`, `disciplines`, and `concepts`.

## Usage

```bash
python src/evaluate_predictions.py \
  --gold benchmark/ground_truth.jsonl.gz \
  --predictions path/to/predictions.jsonl.gz \
  --output outputs/evaluation_metrics.json
```

`--concept-similarity` defaults to `benchmark/concept_similarity.npz`. When both files contain DOI fields, the evaluator aligns predictions to gold rows by DOI.

## Why the matrix is frozen

The taxonomy in `taxonomy/` is the current PhySH vocabulary and may include concepts added after the benchmark was constructed. Pinning to the released 3,793-concept artifact keeps benchmark scores comparable to the paper.
