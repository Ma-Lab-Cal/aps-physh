# PhySH Taxonomy Sources

The taxonomy in `taxonomy/` covers the PhySH discipline and concept IDs observed in the APS metadata corpus through 2025.

## Files

- `taxonomy/facets.json`: facet ID → facet label.
- `taxonomy/disciplines.json`: discipline ID → discipline label.
- `taxonomy/concepts.json`: concept ID → `{label, facet, technique_subfacet?}`.

## High-level facets

Five high-level facets: `Physical Systems`, `Properties`, `Research Areas`, `Techniques`, and `Professional Topics`.

APS metadata also includes narrower technique groupings under `Techniques`: `Experimental Techniques`, `Theoretical Techniques`, `Computational Techniques`, and `Theoretical & Computational Techniques`. These appear as `technique_subfacet` values in `taxonomy/concepts.json` but are rolled up to the single `Techniques` facet in the analysis-facing `facet` field.

## Coverage

The mapping covers every PhySH discipline and concept ID observed in APS metadata through 2025 (18 disciplines, 3,695 concept IDs).

## Current taxonomy versus benchmark inventory

The frozen benchmark similarity matrix has 3,793 concepts. The current taxonomy in `taxonomy/concepts.json` is broader because it includes newer concepts observed through 2025. Use the frozen matrix when reproducing the paper's Semantic F1 scores; use the current taxonomy when running new classification.

## Plot palette

The fixed discipline color palette is not a taxonomy file. It lives in `style/discipline_colors.json` and is used only for discipline-colored figures.
