# PhySH Classification Runs

Five independent PhySH classification runs over the APS research-article corpus. Each file is gzipped JSONL, one row per paper:

```json
{
  "doi": "10.1103/PhysRevSeriesI.1.1",
  "title": "A Study of the Transmission Spectra of Certain Substances in the Infra-Red",
  "predicted_disciplines": ["Atomic, Molecular & Optical"],
  "predicted_concepts": ["Optical properties", "Optical absorption spectroscopy", "Infrared spectroscopy"]
}
```

`run_1.jsonl.gz` through `run_5.jsonl.gz` cover the same paper set with the same model and configuration, but independently resampled few-shot exemplars. Existing APS PhySH labels are retained where present. Discipline and concept trajectories in the paper average across the five runs.

Ground-truth PhySH labels for post-2016 APS articles are in `../benchmark/ground_truth.jsonl.gz` (`doi`, `title`, `disciplines`, `concepts`).

## Loading

```python
import gzip
import json

with gzip.open("runs/run_1.jsonl.gz", "rt", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
```

`pandas` reads gzipped JSONL directly:

```python
import pandas as pd

df = pd.read_json("runs/run_1.jsonl.gz", lines=True, compression="gzip")
```

## Fields

Each row carries only:

- `doi`: APS DOI.
- `title`: canonical article title from APS metadata, HTML stripped and Unicode normalized.
- `predicted_disciplines`: PhySH discipline labels in the order returned by the classifier.
- `predicted_concepts`: PhySH concept labels in the order returned by the classifier.

No other bibliographic, author, abstract, or full-text content is included. APS retains all rights to the underlying article text; see `../NOTICE`.

## License

Files in this directory are released under CC BY 4.0 (`../DATA_LICENSE`), subject to the attribution conditions in `../NOTICE`. The rest of the repository is under the MIT License (`../LICENSE`).
