# Plotting Scripts

Scripts for reproducing the paper's plots from released labels and user-supplied metadata.

- `plot_label_coverage.py`: Fig. 1-style yearly coverage of no subject headings, PACS-era labels, and PhySH labels.
- `plot_discipline_trends.py`: Fig. 4-style discipline share and split-count stacked-bar plots.
- `plot_embedding.py`: discipline-colored UMAP scatter from coordinates and PhySH labels.
- `plot_concept_trends.py`: Fig. 5 concept-trajectory layout from the five released runs and DOI/year metadata.

The released run files carry DOI, title, and PhySH labels only, so the time-series plots require a separate metadata file with DOI and publication year.

For the paper-scale discipline-colored embedding, use `../embedding/umap_default_coordinates.csv.gz` — it covers the released run corpus and the coordinate-available ground-truth papers, generated from `gemini-embedding-2` title-plus-abstract embeddings after `StandardScaler` preprocessing.
