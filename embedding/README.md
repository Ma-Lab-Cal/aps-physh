# Embedding Coordinates

- `umap_default_coordinates.csv.gz`: 2D UMAP coordinates for the paper-scale discipline map. One row per article in the released run corpus or the post-2016 ground-truth corpus, with columns `doi`, `x`, and `y`. 760,888 rows total.

Coordinates were generated from `gemini-embedding-2` title-plus-abstract embeddings after `StandardScaler` preprocessing, followed by UMAP with the paper settings. Abstracts and embedding vectors are not distributed.
