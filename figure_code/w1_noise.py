#!/usr/bin/env python3
"""W1b+W1c: discipline error model from the run∩gold overlap, then noise injection.

Error model (pooled over 5 public runs on the papers carrying both author and LLM
labels -- public runs ∩ canonical gold):
  precision_d = P(d in author labels | d predicted)
  recall_d    = P(d predicted | d in author labels)
  conf[d]     = empirical distribution of author disciplines on false-positive
                instances of d (what the truth was when the model wrongly said d)

Injection (100 replicates): for every LLM-labeled paper in the trajectory corpus,
each predicted discipline d is kept with prob precision_d, else replaced by a draw
from conf[d]; author-labeled papers are untouched. Discipline share curves are
recomputed per replicate; headline features of Fig. 4 are tracked.

Inputs: data/public_runs/ (581,506 rows/run, identical to the public repository),
analysis/gold_canonical.jsonl.gz, analysis/canonical_corpus_rawverified.json (v3).
"""
import gzip, json, sqlite3, collections
import numpy as np

rng = np.random.default_rng(20260731)
ROOT = "."
YEARS = list(range(1893, 2026)); YIDX = {y: i for i, y in enumerate(YEARS)}
TYPES4 = {"article", "rapid", "brief", "letter"}
disc_labels = sorted({l.strip() for l in open(f"{ROOT}/analysis/vocab_disciplines.txt")})
DIDX = {d: i for i, d in enumerate(disc_labels)}; ND = 17

db = sqlite3.connect(f"{ROOT}/data/aps_index.sqlite")
meta = {d: (y, t) for d, y, t in db.execute("SELECT doi, year, article_type FROM papers")}

# canonical gold: fixes the trajectory population AND defines the dual-label
# overlap the error model is fit on
gold_can = {}
gold_can_year = {}
with gzip.open(f"{ROOT}/analysis/gold_canonical.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        gold_can[r["doi"]] = [d for d in (r.get("disciplines") or []) if d in DIDX]
        gold_can_year[r["doi"]] = r.get("year")

_led = json.load(open(f"{ROOT}/analysis/canonical_corpus_rawverified.json"))
EXCLUDED = set(_led["bad_title_dois"]) | {_led["filter"]["doi_not"]}

# ---- W1b: pooled confusion stats over all 5 runs
tp = np.zeros(ND); fp = np.zeros(ND); fn = np.zeros(ND)
conf = np.zeros((ND, ND))  # conf[pred, true] counts on false-positive instances
n_dual = set()
run1_rows = []  # (doi, [disc_idx]) for run_1 non-gold corpus papers (injection base)
for run in range(1, 6):
    with gzip.open(f"{ROOT}/repo_v2/runs/run_{run}.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            doi = r["doi"]
            pred = [d for d in (r.get("predicted_disciplines") or []) if d in DIDX]
            if run == 1 and doi not in gold_can and doi not in EXCLUDED:
                y, t = meta.get(doi, (0, ""))
                if t in TYPES4 and y in YIDX and pred:
                    run1_rows.append((YIDX[y], [DIDX[d] for d in pred]))
            # error model is fit on the 2016 rollout cohort only (Sec. S2/S7)
            g = gold_can.get(doi) if gold_can_year.get(doi) == 2016 else None
            if g is not None: n_dual.add(doi)
            if g is None:
                continue
            gs = set(g); ps = set(pred)
            for d in ps & gs: tp[DIDX[d]] += 1
            for d in ps - gs:
                fp[DIDX[d]] += 1
                tgt = list(gs - ps) or list(gs)
                for t2 in tgt: conf[DIDX[d], DIDX[t2]] += 1.0 / max(len(tgt), 1)
            for d in gs - ps: fn[DIDX[d]] += 1

precision = tp / np.maximum(tp + fp, 1)
recall = tp / np.maximum(tp + fn, 1)
conf_p = conf / np.maximum(conf.sum(1, keepdims=True), 1e-9)
print(f"=== W1b: discipline error model (pooled 5 runs x {len(n_dual):,} overlap papers) ===")
print(f"{'discipline':40s} {'prec':>6s} {'rec':>6s} {'n_gold':>8s}")
for i, d in enumerate(disc_labels):
    print(f"{d[:40]:40s} {precision[i]:6.3f} {recall[i]:6.3f} {int(tp[i]+fn[i]):8d}")
mprec = tp.sum() / max(tp.sum() + fp.sum(), 1)
mrec = tp.sum() / max(tp.sum() + fn.sum(), 1)
print(f"micro precision {mprec:.3f}  micro recall {mrec:.3f}")

np.savez_compressed(f"{ROOT}/analysis/w1_error_model.npz",
                    disciplines=np.array(disc_labels), precision=precision, recall=recall,
                    conf_p=conf_p, tp=tp, fp=fp, fn=fn)

# ---- gold (fixed) contribution to weights: canonical gold, letters included
gd_w = np.zeros((len(YEARS), ND)); n_tot = np.zeros(len(YEARS))
for doi, ds in gold_can.items():
    y = gold_can_year.get(doi)
    if y in YIDX:
        n_tot[YIDX[y]] += 1
        if ds:
            for d in ds: gd_w[YIDX[y], DIDX[d]] += 1.0 / len(ds)
for yi, labs in run1_rows: n_tot[yi] += 1

# flatten run_1 labels for vectorized injection
pair_year = []; pair_disc = []; pair_w = []
for yi, labs in run1_rows:
    w = 1.0 / len(labs)
    for d in labs:
        pair_year.append(yi); pair_disc.append(d); pair_w.append(w)
pair_year = np.array(pair_year); pair_disc = np.array(pair_disc); pair_w = np.array(pair_w)
print(f"\ninjection base: {len(run1_rows)} LLM-labeled papers, {len(pair_disc)} label instances")

def roll3(a):
    pad = np.concatenate([a[:1], a, a[-1:]], axis=0)
    return (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0

def shares_from(pd_arr):
    w = gd_w.copy()
    np.add.at(w, (pair_year, pd_arr), pair_w)
    return roll3(w) / roll3(np.maximum(n_tot, 1))[:, None] * 100.0

def features(sh):
    NP = DIDX["Nuclear Physics"]; CMP = DIDX["Condensed Matter, Materials & Applied Physics"]
    QIS = DIDX["Quantum Information, Science & Technology"]; AMO = DIDX["Atomic, Molecular & Optical"]
    rank1 = sh.argmax(1)
    np_first = next((YEARS[i] for i in range(len(YEARS)) if rank1[i] == NP), None)
    cmp_last = None
    for i in range(len(YEARS)):
        if rank1[i] == CMP and all(rank1[j] == CMP for j in range(i, min(i + 10, len(YEARS)))):
            cmp_last = YEARS[i]; break
    qis1 = next((YEARS[i] for i in range(len(YEARS)) if sh[i, QIS] >= 1.0), None)
    return {
        "np_first_rank1": np_first,
        "np_war_dip": float(sh[YIDX[1944], NP] - sh[YIDX[1941], NP]),
        "cmp_rank1_stable": cmp_last,
        "qis_cross_1pct": qis1,
        "amo_1930": float(sh[YIDX[1930], AMO]),
        "np_1939": float(sh[YIDX[1939], NP]),
        "cmp_1980": float(sh[YIDX[1980], CMP]),
    }

base_sh = shares_from(pair_disc)
base_feat = features(base_sh)
print("\nbase features (run_1 + gold):", json.dumps(base_feat))

# precompute replacement tables
keep_p = precision[pair_disc]
REPS = 100
all_sh = np.empty((REPS, len(YEARS), ND), dtype=np.float32)
feats = []
cum = conf_p.cumsum(1)
for rep in range(REPS):
    keep = rng.random(len(pair_disc)) < keep_p
    new_disc = pair_disc.copy()
    idx_bad = np.where(~keep)[0]
    if len(idx_bad):
        u = rng.random(len(idx_bad))
        rows = cum[pair_disc[idx_bad]]
        new_disc[idx_bad] = (u[:, None] < rows).argmax(1)
    sh = shares_from(new_disc)
    all_sh[rep] = sh
    feats.append(features(sh))
    if (rep + 1) % 25 == 0: print(f"  replicate {rep+1}/{REPS}")

np.savez_compressed(f"{ROOT}/analysis/w1_noise_replicates.npz",
                    years=np.array(YEARS), disciplines=np.array(disc_labels),
                    base=base_sh.astype(np.float32), reps=all_sh)

print("\n=== W1c: feature stability under injected noise (100 replicates) ===")
for k in base_feat:
    vals = [f[k] for f in feats if f[k] is not None]
    if isinstance(base_feat[k], float):
        print(f"{k:20s} base {base_feat[k]:7.2f} | reps mean {np.mean(vals):7.2f} sd {np.std(vals):5.2f} range [{min(vals):.2f},{max(vals):.2f}]")
    else:
        cnt = collections.Counter(vals)
        print(f"{k:20s} base {base_feat[k]} | reps mode {cnt.most_common(1)[0]} distinct {sorted(cnt)}")

dev = np.abs(all_sh - base_sh[None]).astype(float)
print(f"\nshare deviation from base: mean {dev.mean():.3f} pp | p95 {np.percentile(dev,95):.3f} pp | max {dev.max():.3f} pp")
post1920 = slice(YIDX[1920], None)
print(f"post-1920 only:            mean {dev[:,post1920].mean():.3f} pp | p95 {np.percentile(dev[:,post1920],95):.3f} pp | max {dev[:,post1920].max():.3f} pp")
