#!/usr/bin/env python3
"""W1a: per-run discipline share/count trajectories + Fig-5 concept trajectories.

Reproduces the paper's aggregation on the canonical corpus (filter v3, see
analysis/canonical_corpus_rawverified.json) from the PUBLIC run files
(data/public_runs/, 581,506 rows per run -- identical to the repository at
github.com/Ma-Lab-Cal/aps-physh): author-assigned PhySH labels take precedence
where they exist (identical across runs, from analysis/gold_canonical.jsonl.gz
-- includes letters); LLM labels otherwise (vary across runs); run rows outside
the canonical corpus are dropped via the sqlite index. Discipline shares use
equal fractional weights for multi-discipline papers; concept trajectories use
share-of-papers-tagged. Three-year centered rolling averages, as in Figs. 4-5.
"""
import gzip, json, sqlite3, collections
import numpy as np

ROOT = "."
YEARS = list(range(1893, 2026))
YIDX = {y: i for i, y in enumerate(YEARS)}
TYPES4 = {"article", "rapid", "brief", "letter"}
_led = json.load(open(f"{ROOT}/analysis/canonical_corpus_rawverified.json"))
EXCLUDED = set(_led["bad_title_dois"]) | {_led["filter"]["doi_not"]}

CONCEPTS = [
    "Atomic spectra", "Nuclear reactions", "Mesons", "Quantum chromodynamics",
    "X-ray diffraction", "Deep inelastic scattering", "Cosmic microwave background", "Graphene",
    "Renormalization group", "Perturbation theory", "Green's function methods",
    "Density functional theory", "Monte Carlo methods", "Tensor network methods",
    "Topological insulators", "Quantum information processing", "Machine learning",
    "Gravitational wave sources", "Spintronics",
]

db = sqlite3.connect(f"{ROOT}/data/aps_index.sqlite")
meta = {d: (y, t) for d, y, t in db.execute("SELECT doi, year, article_type FROM papers")}

gold = {}
gold_year = {}
with gzip.open(f"{ROOT}/analysis/gold_canonical.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        gold[r["doi"]] = (r.get("disciplines") or [], r.get("concepts") or [])
        gold_year[r["doi"]] = r.get("year")

disc_labels = sorted({l.strip() for l in open(f"{ROOT}/analysis/vocab_disciplines.txt")})
DIDX = {d: i for i, d in enumerate(disc_labels)}
CIDX = {c: i for i, c in enumerate(CONCEPTS)}

def roll3(a, axis=0):
    """3-year centered rolling mean along `axis` (edges use available window)."""
    pad = np.concatenate([a[:1], a, a[-1:]], axis=0) if axis == 0 else None
    return (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0

# gold contribution (identical across runs); every canonical-gold row is a
# corpus member by construction, letters included
gd_w = np.zeros((len(YEARS), 17))
gd_tag = np.zeros((len(YEARS), len(CONCEPTS)))
gd_n = np.zeros(len(YEARS))
for doi, (ds, cs) in gold.items():
    y = gold_year.get(doi)
    if y not in YIDX:
        continue
    yi = YIDX[y]
    gd_n[yi] += 1
    ds_v = [d for d in ds if d in DIDX]
    for d in ds_v:
        gd_w[yi, DIDX[d]] += 1.0 / len(ds_v)
    for c in set(cs):
        if c in CIDX:
            gd_tag[yi, CIDX[c]] += 1

shares, counts, ctraj = [], [], []
for run in range(1, 6):
    w = gd_w.copy(); tag = gd_tag.copy(); n = gd_n.copy()
    with gzip.open(f"{ROOT}/repo_v2/runs/run_{run}.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            doi = r["doi"]
            if doi in gold:
                continue  # author labels take precedence
            if doi in EXCLUDED:
                continue  # canonical filter v3: reviewed non-research + 1949 letter
            y, t = meta.get(doi, (0, ""))
            if t not in TYPES4 or y not in YIDX:
                continue
            yi = YIDX[y]
            n[yi] += 1
            ds_v = [d for d in r.get("predicted_disciplines") or [] if d in DIDX]
            for d in ds_v:
                w[yi, DIDX[d]] += 1.0 / len(ds_v)
            for c in set(r.get("predicted_concepts") or []):
                if c in CIDX:
                    tag[yi, CIDX[c]] += 1
    n_safe = np.maximum(n, 1)
    shares.append(roll3(w) / roll3(n_safe)[:, None] * 100.0)
    counts.append(roll3(w))
    ctraj.append(roll3(tag) / roll3(n_safe)[:, None] * 100.0)
    print(f"run {run}: corpus papers {int(n.sum())}")

shares = np.array(shares); counts = np.array(counts); ctraj = np.array(ctraj)
np.savez_compressed(
    f"{ROOT}/analysis/w1_trajectories.npz",
    years=np.array(YEARS), disciplines=np.array(disc_labels), concepts=np.array(CONCEPTS),
    shares=shares, counts=counts, concept_shares=ctraj, gold_n=gd_n,
)

# band-width summary (discipline shares, %)
band = shares.max(0) - shares.min(0)
mean_share = shares.mean(0)
print("\n=== Run-to-run band widths, discipline shares (percentage points) ===")
print(f"mean band width over all year-discipline cells: {band.mean():.3f} pp")
print(f"95th percentile: {np.percentile(band, 95):.3f} pp | max: {band.max():.3f} pp")
flat = [(band[yi, di], YEARS[yi], disc_labels[di], mean_share[yi, di])
        for yi in range(len(YEARS)) for di in range(17)]
flat.sort(reverse=True)
print("largest bands (band pp, year, discipline, mean share):")
for b, y, d, m in flat[:8]:
    print(f"  {b:5.2f}  {y}  {d[:38]:38s} mean {m:5.2f}%")

cband = ctraj.max(0) - ctraj.min(0)
print("\n=== Concept trajectory band widths (pp) ===")
for ci, c in enumerate(CONCEPTS):
    print(f"  {c[:36]:36s} peak {ctraj.mean(0)[:, ci].max():6.3f}%  mean band {cband[:, ci].mean():6.4f}  max band {cband[:, ci].max():6.4f}")
