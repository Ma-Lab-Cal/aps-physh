# Benchmarking and Semantic Similarity Metrics

Evaluation procedure for model selection and labeling-parameter optimization (Fig. 3), corresponding to Appendix B of "A Unified Subject Map for 130 Years of Physics."

## Benchmark data

Evaluation used a held-out set of post-2016 APS papers with author-assigned PhySH labels, disjoint from the historical pre-2016 labeling target. Each benchmark record carries title, abstract, gold PhySH disciplines, and gold PhySH concepts.

The abstract-bearing held-out records are not redistributed. A DOI/title-only ground-truth file for 197,167 post-2016 APS research articles with existing PhySH labels is at `benchmark/ground_truth.jsonl.gz`; each row has `doi`, `title`, `disciplines`, and `concepts`.

Reported numbers:

- Concept Semantic F1: 80.9% on a 1,000-paper held-out benchmark.
- Discipline F1: 77.6% on the same 1,000-paper held-out benchmark.

The same evaluator drives the model comparison in Fig. 3(a) and the few-shot-count/batch-size grid search in Fig. 3(b,c). The production run then labeled the full historical archive five times with resampled in-context examples; those runs are in `runs/`.

## Discipline metric

Disciplines are scored by exact sample F1. For paper `i`, with predicted set `D_i` and gold set `G_i`:

```text
F1_i = 2 |D_i ∩ G_i| / (|D_i| + |G_i|)
```

The reported score is the arithmetic mean over benchmark papers. Strict by design — the discipline vocabulary is small and categorical.

## Concept metric: sample Semantic F1

Concepts are scored by sample Semantic F1, following Chochlakis et al. (2025), because the concept vocabulary is large and semantically graded. A prediction that misses the exact human concept but lands on a closely related PhySH concept receives partial credit according to a fixed concept-similarity matrix.

For paper `i`, let `C_i` be the predicted concept set and `T_i` the gold set. Semantic precision and recall:

```text
P_i = (1 / |C_i|) sum_{c in C_i} max_{t in T_i} S[c,t]
R_i = (1 / |T_i|) sum_{t in T_i} max_{c in C_i} S[t,c]
```

Per-paper semantic F1 is the harmonic mean:

```text
F1_i_sem = 2 P_i R_i / (P_i + R_i)
```

The reported concept Semantic F1 is the mean of `F1_i_sem` over papers. When `S` is the identity matrix, this reduces to exact sample F1.

## Frozen benchmark matrix

`benchmark/concept_similarity.npz` is consumed by `src/evaluate_predictions.py`. `benchmark/ground_truth.jsonl.gz` is the DOI/title-only gold label file.

The public taxonomy in `taxonomy/concepts.json` may contain newer concepts. Reproduce the paper's benchmark scores with the frozen 3,793-concept artifact, not the current taxonomy.

## Concept similarity construction

We constructed `S` from dense Gemini embeddings of canonical PhySH concept names. Pairwise cosine similarities were mapped to `[0, 1]` and globally rescaled so that exact matches have similarity 1 and the least-related pair has similarity 0. Nearby PhySH concepts get partial credit; exact agreement remains the highest possible score.

The repository ships the frozen matrix but not the construction scripts — this ties benchmark reproduction to the released artifact rather than to later changes in model versions or taxonomy files.

## Scoring model predictions

```bash
python src/classify_batch.py \
  --input path/to/benchmark_gold.jsonl \
  --examples path/to/few_shot_examples.jsonl \
  --model gemini-2.5-flash-lite-preview-09-2025 \
  --output outputs/model_predictions.jsonl

python src/evaluate_predictions.py \
  --gold path/to/benchmark_gold.jsonl \
  --predictions outputs/model_predictions.jsonl \
  --output outputs/evaluation_metrics.json
```

Change `--model` to compare candidates. The optional `--examples` file should be JSONL with title/abstract text and gold PhySH labels.

## Running the evaluator only

```bash
python src/evaluate_predictions.py \
  --gold benchmark/ground_truth.jsonl.gz \
  --predictions path/to/predictions.jsonl.gz \
  --output outputs/evaluation_metrics.json
```

`--concept-similarity` defaults to `benchmark/concept_similarity.npz`.

When both files contain DOI fields, the evaluator aligns by DOI. Gold rows may use `disciplines`/`concepts`, `gold_disciplines`/`gold_concepts`, `true_disciplines`/`true_concepts`, or APS-style `classificationSchemes.physh` entries. Prediction rows should use `predicted_disciplines` and `predicted_concepts`.

## Parameter optimization

The grid search in Fig. 3(b,c) varied in-context example count and papers per batch, scoring each configuration on the same held-out set. Selected setting:

- Model: Gemini 2.5 Flash Lite Preview (`gemini-2.5-flash-lite-preview-09-2025`).
- Few-shot examples: 1,000.
- Papers per batch: 50.
- Temperature: 0.2.
- Top-p: 0.95.
- Top-k: 1.

The full archive was then labeled in five independent runs with resampled example sets. Released runs are in `runs/run_1.jsonl.gz` through `run_5.jsonl.gz`.
