# PhySH Taxonomy

Three files defining the label vocabulary for the public scripts.

## Files

- `facets.json`: facet ID → facet label. Five high-level facets: `Physical Systems`, `Properties`, `Research Areas`, `Techniques`, and `Professional Topics`.
- `disciplines.json`: discipline ID → discipline label. 17 entries.
- `concepts.json`: concept ID → `{label, facet, technique_subfacet?}`.
  - `label`: canonical PhySH concept label.
  - `facet`: one of the five high-level facets above.
  - `technique_subfacet`: optional narrower grouping (`Experimental Techniques`, `Theoretical Techniques`, `Computational Techniques`, or `Theoretical & Computational Techniques`). The paper analyses roll these up to the single `Techniques` facet.

## Scope

These files cover every PhySH discipline and concept ID observed in APS metadata through 2025. They supply the allowed-label lists for `src/classify_one_paper.py`, `src/classify_batch.py`, and the evaluation helpers in `src/physh_tools/`.

To reproduce the paper's benchmark scores exactly, use the frozen 3,793-concept matrix in `benchmark/` — the current taxonomy is broader.
