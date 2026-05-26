# A Unified Subject Map for 130 Years of Physics

Code and data for the paper **"A Unified Subject Map for 130 Years of Physics"** by Khoa Nguyen, Pragyan Pandey, Sophie Li, and Eric Y. Ma (2026). Contact: Eric Y. Ma (eric.y.ma@berkeley.edu). Canonical repository: https://github.com/Ma-Lab-Cal/aps-physh.

Five independent classification runs over the 1893–2025 APS archive are included, along with scripts for PhySH classification, embedding, UMAP reduction, benchmarking, and figure reproduction. All scripts accept user-provided title/abstract data.

## Layout

```text
src/                            # Python scripts and the physh_tools package
plotting/                       # Figure and visualization scripts
taxonomy/                       # Current PhySH taxonomy (3 files)
  facets.json                   #   5 high-level facets
  disciplines.json              #   17 disciplines
  concepts.json                 #   concept ID -> {label, facet, technique_subfacet?}
benchmark/                      # Frozen artifacts for paper-reproducible Semantic F1
  concept_similarity.npz        #   3,793 x 3,793 matrix consumed by the evaluator
  ground_truth.jsonl.gz        #   DOI/title plus APS PhySH labels
embedding/
  umap_default_coordinates.csv.gz  # default 2D UMAP coordinates for released + gold papers
runs/                           # Five PhySH classification runs of the APS archive
  run_1.jsonl.gz ... run_5.jsonl.gz  # five independent runs with resampled few-shot exemplars
style/
  discipline_colors.json        # Fixed discipline color palette for plots
docs/                           # Method notes for the supplement
examples/                       # Tiny demo inputs for the runnable scripts
outputs/                        # Default destination for generated files
LICENSE                         # MIT, covers code and taxonomy files
DATA_LICENSE                    # CC BY 4.0, covers files under runs/
NOTICE                          # APS metadata acknowledgement and attribution terms
CITATION.cff                    # How to cite this repository and the accompanying paper
```

See `src/README.md` for computational scripts, `plotting/README.md` for figure scripts, and `taxonomy/README.md`, `benchmark/README.md`, and `runs/README.md` for details on those data directories.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # then edit GOOGLE_API_KEY
```

On macOS/Linux, replace `copy` with `cp`.

Input records may use a simple schema with `title`, `abstract`, and optional `doi`/`year`, or an APS-style schema with `title.value`, `abstract.value`, and `identifiers.doi`.

## Common Workflows

### Classify one paper

```bash
python src/classify_one_paper.py \
  --title "Topological edge modes in a driven photonic lattice" \
  --abstract "We study protected edge transport in a periodically driven photonic lattice and characterize the resulting band topology." \
  --output outputs/one_paper_classification.json
```

The output JSON contains `predicted_disciplines` and `predicted_concepts`. The model is constrained to the PhySH label lists in `taxonomy/`.

### Embed one paper

Demonstrates the `gemini-embedding-2` call used in the paper. A single paper can be embedded but not meaningfully UMAP-reduced alone.

```bash
python src/embed_one_paper.py \
  --title "Topological edge modes in a driven photonic lattice" \
  --abstract "We study protected edge transport in a periodically driven photonic lattice and characterize the resulting band topology." \
  --output-dir outputs/one_paper_embedding
```

Writes `embedding.npy` and `metadata.json`.

### Embed and reduce a corpus

```bash
python src/embed_papers.py \
  --input examples/sample_papers.jsonl \
  --output-dir outputs/sample_embeddings

python src/reduce_umap.py \
  --manifest outputs/sample_embeddings/manifest.csv \
  --embeddings outputs/sample_embeddings/embeddings.npy \
  --output outputs/sample_umap.csv
```

Default UMAP parameters match the paper: `StandardScaler` followed by UMAP with `n_neighbors=15`, `min_dist=0.1`, and two output components.

### Classify a corpus

```bash
python src/classify_batch.py \
  --input examples/sample_papers.jsonl \
  --output outputs/sample_labels.jsonl
```

The paper-scale run used 1,000 in-context examples, batches of 50 papers, and five independent resamplings. The same prompt structure and label constraints are exposed here for user-provided data.

### Plot a discipline-colored embedding

```bash
python plotting/plot_embedding.py \
  --coordinates embedding/umap_default_coordinates.csv.gz \
  --labels runs/run_1.jsonl.gz \
  --output outputs/fig_umap_disciplines.png
```

Coordinates are joined to labels by DOI when available, otherwise by local `paper_id`. Points are colored by the first predicted discipline.

For the paper-scale map, use `embedding/umap_default_coordinates.csv.gz` as the coordinate input — it covers the released corpus plus the coordinate-available ground-truth papers.

### Plot discipline trends

Fig. 4's discipline time-series plots need the released runs and a DOI/year metadata file:

```bash
python plotting/plot_discipline_trends.py \
  --runs runs/run_1.jsonl.gz runs/run_2.jsonl.gz runs/run_3.jsonl.gz runs/run_4.jsonl.gz runs/run_5.jsonl.gz \
  --metadata path/to/aps_metadata_with_years.jsonl \
  --output-share outputs/fig_discipline_share.png \
  --output-counts outputs/fig_discipline_counts_split.png \
  --output-legend outputs/fig_discipline_legend.png
```

### Plot concept trajectories

Reproduces the Fig. 5 layout: four panels, three-year smoothing, five-run consensus, and discovery/Nobel markers. Needs the released runs and a DOI/year metadata file:

```bash
python plotting/plot_concept_trends.py \
  --runs runs/run_1.jsonl.gz runs/run_2.jsonl.gz runs/run_3.jsonl.gz runs/run_4.jsonl.gz runs/run_5.jsonl.gz \
  --metadata path/to/aps_metadata_with_years.jsonl \
  --output outputs/fig_concept_trends.png
```

The released run files contain DOI, title, and PhySH labels only; a separate metadata file supplies publication years for the time-series plots.

### Plot label coverage

Fig. 1-style coverage plot:

```bash
python plotting/plot_label_coverage.py \
  --metadata path/to/aps_metadata_2025.jsonl \
  --output-png outputs/fig_label_coverage_metadata_2025.png \
  --output-svg outputs/fig_label_coverage_metadata_2025.svg
```

## Reproduce Benchmark Metrics

We report exact sample F1 for disciplines and Semantic F1 for concepts. The frozen 3,793-concept similarity matrix and DOI/title-only labels for 197,167 post-2016 articles are in `benchmark/`.

To score model predictions against the included ground truth:

```bash
python src/evaluate_predictions.py \
  --gold benchmark/ground_truth.jsonl.gz \
  --predictions path/to/predictions.jsonl.gz \
  --output outputs/evaluation_metrics.json
```

The evaluator aligns rows by DOI. Generate predictions with `src/classify_batch.py`, then pass them to the evaluator.

To run the evaluator without an API key, use the synthetic demo:

```bash
python src/evaluate_predictions.py \
  --gold examples/benchmark_demo_gold.jsonl \
  --predictions examples/benchmark_demo_predictions.jsonl \
  --output outputs/evaluation_demo_metrics.json
```

For details on the metrics and similarity construction, see `docs/benchmarking_and_metrics.md`.

## Released Data

Five PhySH classification runs are under `runs/`:

```python
import pandas as pd
df = pd.read_json("runs/run_1.jsonl.gz", lines=True, compression="gzip")
```

Prediction rows carry only `doi`, `title`, `predicted_disciplines`, and `predicted_concepts`. Gold rows in `benchmark/` store labels as `disciplines` and `concepts`. The UMAP file has DOI and 2D coordinates only. No abstracts, full text, or author lists are included; APS retains rights to that content. See `runs/README.md`, `benchmark/README.md`, `embedding/README.md`, and `NOTICE`.

To run on your own corpus, provide a JSONL with title and abstract fields and follow the `Embed and reduce a corpus` recipe above.

## APS Metadata Access

The full APS metadata corpus is not redistributed here. Open-access metadata is at `https://harvest.aps.org` (no key required). Full Harvest API access typically requires a separate APS agreement by IP address; contact `customercare@aps.org` to start that process.

## License and Citation

- Code, taxonomy mappings, plotting style, benchmark scripts, docs, and examples: MIT License (`LICENSE`).
- Files under `runs/`, `benchmark/ground_truth.jsonl.gz`, and `embedding/umap_default_coordinates.csv.gz`: CC BY 4.0 (`DATA_LICENSE`), subject to the APS attribution notice in `NOTICE`.
- Citation metadata: `CITATION.cff`. Please cite the paper "A Unified Subject Map for 130 Years of Physics" by Nguyen, Pandey, Li, and Ma (2026). The final DOI will be filled in after publication.
