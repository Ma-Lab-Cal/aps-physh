# Embedding, UMAP reduction, and discipline coloring

Settings for the embedding view in Fig. 4(c) of "A Unified Subject Map for 130 Years of Physics." UMAP coordinates are computed from title and abstract text only; PhySH discipline labels are overlaid afterward for coloring.

## Text representation

Each paper is represented by its title and abstract. HTML tags are stripped, Unicode is normalized, and whitespace is collapsed. The scripts accept either a simple schema (`title`, `abstract`, `doi`, `year`) or APS-style records (`title.value`, `abstract.value`, `identifiers.doi`).

The paper-scale analysis covered 756,183 APS papers from 1893 through 2025.

## Embedding model

We used the Gemini Embedding model (`gemini-embedding-2`) with 768 output dimensions. Each text was prefixed as:

```text
task: clustering | query: {title + abstract}
```

Embeddings were stored as `float32`. After each API batch, vectors were L2-normalized row-wise. This normalized semantic space was the input to dimensionality reduction.

## UMAP reduction

The full embedding matrix was standardized dimension-wise with `sklearn.preprocessing.StandardScaler`, then reduced with CPU `umap-learn`:

- `n_components = 2`
- `n_neighbors = 15`
- `min_dist = 0.1`
- Euclidean metric on the scaled vectors

Output columns: `doi`, `paper_id`, `x`, `y`.

## Discipline coloring

Coordinates are joined to labels by DOI when possible, otherwise by `paper_id`. The plotted color is the first entry in `predicted_disciplines`, treated as the primary PhySH discipline.

The color palette is fixed per discipline label and lives in `style/discipline_colors.json`, keeping the embedding plot consistent with companion time-series figures. Unknown disciplines are colored light gray.

## Plot rendering

The public script uses a square figure, small semi-transparent points, no axes, and no legend by default. Points are shuffled before plotting so one discipline does not systematically cover another. Axis limits are cropped to the 1st–99th percentile range with small padding to suppress extreme outliers while leaving coordinates unchanged.

## Interpretation

The UMAP layout is an unsupervised semantic projection of title and abstract embeddings. Discipline colors are overlaid after the projection. Spatial separation reflects geometry in embedding space; color coherence indicates alignment between that geometry and the PhySH discipline taxonomy.
