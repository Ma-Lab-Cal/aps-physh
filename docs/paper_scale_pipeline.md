# Paper-scale pipeline

How the paper-scale classification was run and how the public scripts relate to it.

## Classification setup

We classified APS titles and abstracts into official PhySH disciplines and concepts using Gemini 2.5 Flash Lite Preview (API identifier `gemini-2.5-flash-lite-preview-09-2025`).

## Corpus filtering

We built the corpus from APS metadata records whose `articleType` was `article`, `rapid`, or `brief`.

For papers with existing APS-assigned PhySH labels, those labels were retained. LLM labeling was applied to papers without PhySH metadata.

The prompt had four parts:

1. Task instructions requiring canonical PhySH labels.
2. The complete valid PhySH discipline and concept lists.
3. Curated PhySH-labeled in-context examples.
4. A batch of papers represented by title and abstract.

Selected configuration:

- 1,000 in-context examples from modern PhySH-labeled APS papers.
- 50 papers per inference batch.
- Temperature: 0.2.
- Top-p: 0.95.
- Top-k: 1.
- Structured JSON output with `disciplines` and `concepts` lists per paper.

In-context examples were drawn from modern APS research articles with author-assigned PhySH labels. Each example carried title-plus-abstract text, one or more PhySH disciplines, and a list of PhySH concepts. The example set was resampled across the five independent runs. The examples themselves are not redistributed because they contain APS abstracts.

To reduce sampling variability in historical trends, the full archive was labeled in five independent runs with resampled example sets. Reported discipline and concept trajectories average over these runs.

## Model selection and parameter optimization

Model selection and labeling-parameter optimization used held-out modern APS papers with existing PhySH labels. Disciplines were scored by exact sample F1. Concepts were scored by sample Semantic F1 using the frozen 3,793-concept dense-embedding similarity matrix in `benchmark/`. See `docs/benchmarking_and_metrics.md` for the formulas and public evaluator.

Gemini 2.5 Flash Lite Preview was chosen because it gave strong concept and discipline scores while remaining fast and cost-effective enough for archive-scale labeling. The final setting used 1,000 in-context examples and 50 papers per batch.

## Public scripts

The public scripts run the same method on user-provided data:

- `src/classify_one_paper.py`: run the PhySH classifier on one input paper.
- `src/classify_batch.py`: run on a JSONL file.
- `src/embed_one_paper.py`: demonstrate the `gemini-embedding-2` call for one paper.
- `src/embed_papers.py`, `src/reduce_umap.py`, and `plotting/plot_embedding.py`: reproduce the embedding, reduction, and coloring workflow on a corpus.

## Released data scope

The five classification runs are in `runs/` as gzipped JSONL. Each row carries only `doi`, `title`, `predicted_disciplines`, and `predicted_concepts`. Abstracts, full text, and other APS-copyrighted content are not redistributed; DOIs and titles are released with APS permission. See `runs/README.md` and `NOTICE`.
