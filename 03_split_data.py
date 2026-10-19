"""
Step 4: Group-aware, multilabel-stratified train / validation / test split.

Why not a random split?
  Sentences from the same malware / group are written in a similar style and describe
  similar behavior. A random split would put near-identical sentences in both train and
  test (data leakage) and inflate the scores. So we split by GROUP (source), and we keep
  the tactic distribution similar across splits (important for rare tactics).

Input : data/clean.csv
Output: data/train.csv, data/val.csv, data/test.csv
"""
import numpy as np
import pandas as pd

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

df = pd.read_csv("data/clean.csv")
df["tactic_list"] = df["tactics"].str.split("|")

# ---------- 1. Build groups ----------
# A sentence can come from several sources. Merge such sources into one group
# (union-find) so that the same sentence can never be split across sets.
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


for srcs in df["sources"].str.split("|"):
    find(srcs[0])
    for s in srcs[1:]:
        union(srcs[0], s)

df["group_id"] = df["sources"].str.split("|").str[0].map(find)
print("Sources -> groups:", df["sources"].str.split("|").explode().nunique(),
      "->", df["group_id"].nunique())

# ---------- 2. Multi-hot tactic matrix ----------
tactics = sorted({t for ts in df["tactic_list"] for t in ts})
Y = np.zeros((len(df), len(tactics)), dtype=int)
for i, ts in enumerate(df["tactic_list"]):
    for t in ts:
        Y[i, tactics.index(t)] = 1

# Per-group label counts (+ one extra column = number of sentences)
group_ids = df["group_id"].unique()
G = {}
for gid, idx in df.groupby("group_id").indices.items():
    G[gid] = np.append(Y[idx].sum(axis=0), len(idx))

total = np.sum(list(G.values()), axis=0).astype(float)   # totals per tactic + size
weights = 1.0 / np.maximum(total, 1)                      # rare tactics matter more
weights[-1] = 1.0 / total[-1]

# ---------- 3. Greedy assignment of groups to splits ----------
rng = np.random.default_rng(SEED)
target = {s: total * r for s, r in RATIOS.items()}
current = {s: np.zeros_like(total) for s in RATIOS}
assignment = {}

# Process groups holding the rarest labels first, then the largest ones
order = list(group_ids)
rng.shuffle(order)
order.sort(key=lambda g: -(G[g][:-1] * weights[:-1]).sum())


def cost(split, vec):
    new = current[split] + vec
    return float((weights * ((new - target[split]) / np.maximum(target[split], 1)) ** 2).sum())


for gid in order:
    best = min(RATIOS, key=lambda s: cost(s, G[gid]) - cost(s, np.zeros_like(total)))
    assignment[gid] = best
    current[best] += G[gid]

df["split"] = df["group_id"].map(assignment)

# ---------- 4. Save ----------
cols = ["text_clean", "tactics", "technique_ids", "has_tech_name", "group_id"]
for s in RATIOS:
    df[df["split"] == s][cols].reset_index(drop=True).to_csv(f"data/{s}.csv", index=False)

# ---------- 5. Report ----------
line = "=" * 60
print(line)
print("SPLIT SUMMARY")
print(line)
sizes = df["split"].value_counts()
for s in RATIOS:
    print(f"{s:<6}: {sizes[s]:>6,} sentences ({sizes[s] / len(df) * 100:4.1f}%) | "
          f"{df[df['split'] == s]['group_id'].nunique():>4} groups")

print("\nTactic distribution (sentences per split):")
print(f"  {'tactic':<22}{'train':>7}{'val':>7}{'test':>7}   test share")
for j, t in enumerate(tactics):
    c = {s: int(Y[(df['split'] == s).values][:, j].sum()) for s in RATIOS}
    tot = sum(c.values())
    print(f"  {t:<22}{c['train']:>7}{c['val']:>7}{c['test']:>7}   {c['test'] / tot * 100:5.1f}%")

# Group-overlap check (must be empty)
sets = {s: set(df[df['split'] == s]['group_id']) for s in RATIOS}
overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
print("\nGroup overlap between splits:", len(overlap), "(must be 0)")

# ---------- 6. Near-duplicate leakage check ----------
from sklearn.feature_extraction.text import TfidfVectorizer

tr = df[df["split"] == "train"]["text_clean"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(tr)
Xtr = vec.transform(tr)
for s in ("val", "test"):
    Xs = vec.transform(df[df["split"] == s]["text_clean"])
    maxsim = np.asarray((Xs @ Xtr.T).max(axis=1).todense()).ravel()
    print(f"{s:<5}: share of sentences with cosine >= 0.9 to a train sentence: "
          f"{(maxsim >= 0.9).mean() * 100:.1f}%   (median max-similarity {np.median(maxsim):.2f})")
print("\nSaved: data/train.csv, data/val.csv, data/test.csv")
