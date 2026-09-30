from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce paper embeddings to 2D with UMAP.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--no-standard-scale", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    vectors = np.load(args.embeddings)
    if len(vectors) != len(manifest):
        raise SystemExit(f"Embedding rows ({len(vectors)}) do not match manifest rows ({len(manifest)}).")
    if len(vectors) < 3:
        raise SystemExit("UMAP needs at least three papers; use embedding output directly for one-paper demos.")

    x = vectors.astype(np.float32, copy=False)
    if not args.no_standard_scale:
        x = StandardScaler().fit_transform(x)

    n_neighbors = min(args.n_neighbors, len(vectors) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=True,
    )
    coords = reducer.fit_transform(x)
    out = manifest[["paper_id", "doi"]].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out.to_csv(args.output, index=False)
    print(f"Wrote UMAP coordinates for {len(out)} papers -> {args.output}")


if __name__ == "__main__":
    main()
