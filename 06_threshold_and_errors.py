"""
Step 7a: Per-tactic threshold tuning (on validation) + error analysis (on validation).

The test set is NOT used here.

Input : data/train.csv, data/val.csv, data/val_probs_bert.npy
Output: data/thresholds_bert.json, data/val_errors_bert.csv
"""
import json
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
mlb = MultiLabelBinarizer().fit(train["tactics"].str.split("|"))
labels = np.array(mlb.classes_)
Y = mlb.transform(val["tactics"].str.split("|"))
P = np.load("data/val_probs_bert.npy")
GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


def predict(P, thr):
    pred = (P >= thr).astype(int)
    empty = pred.sum(axis=1) == 0                      # every sentence has >= 1 tactic
    pred[empty, (P[empty] / thr).argmax(axis=1)] = 1
    return pred


def metrics(Y, pred):
    return (f1_score(Y, pred, average="micro", zero_division=0),
            f1_score(Y, pred, average="macro", zero_division=0),
            float((Y == pred).all(axis=1).mean()))


def tune(P, Y):
    """Best threshold per tactic = the one that maximizes that tactic's F1."""
    thr = np.full(P.shape[1], 0.5)
    for j in range(P.shape[1]):
        scores = [f1_score(Y[:, j], (P[:, j] >= t).astype(int), zero_division=0) for t in GRID]
        thr[j] = GRID[int(np.argmax(scores))]
    return thr


line = "=" * 66
# ---------- 1. Threshold tuning ----------
print(line); print("1. THRESHOLD TUNING (validation)"); print(line)
default = np.full(len(labels), 0.5)
thr_all = tune(P, Y)
m0, m1 = metrics(Y, predict(P, default)), metrics(Y, predict(P, thr_all))
print(f"{'':<34}{'micro-F1':>9}{'macro-F1':>10}{'exact':>8}")
print(f"{'Default threshold 0.5':<34}{m0[0]:>9.4f}{m0[1]:>10.4f}{m0[2]:>8.4f}")
print(f"{'Tuned on all of val (optimistic)':<34}{m1[0]:>9.4f}{m1[1]:>10.4f}{m1[2]:>8.4f}")

# Honest estimate: tune on one half of val, evaluate on the other half (grouped by source group)
gkf = GroupKFold(n_splits=2)
pred_cv = np.zeros_like(Y)
for tr_idx, te_idx in gkf.split(P, Y, groups=val["group_id"]):
    pred_cv[te_idx] = predict(P[te_idx], tune(P[tr_idx], Y[tr_idx]))
m2 = metrics(Y, pred_cv)
print(f"{'Tuned with 2-fold CV (honest)':<34}{m2[0]:>9.4f}{m2[1]:>10.4f}{m2[2]:>8.4f}")

print("\nTuned thresholds:")
for l, t in zip(labels, thr_all):
    print(f"  {l:<22}{t:.2f}")

# Decision: use tuned thresholds only if the honest estimate improves macro-F1
use_tuned = m2[1] > m0[1] + 0.002
final_thr = thr_all if use_tuned else default
print(f"\nDecision: {'USE tuned thresholds' if use_tuned else 'KEEP default 0.5 (tuning does not help in honest CV)'}")
json.dump({l: float(t) for l, t in zip(labels, final_thr)}, open("data/thresholds_bert.json", "w"), indent=2)

pred = predict(P, final_thr)

# ---------- 2. Error analysis ----------
print("\n" + line); print("2. ERROR ANALYSIS (validation, final thresholds)"); print(line)
p, r, f, s = precision_recall_fscore_support(Y, pred, zero_division=0)
print(f"{'tactic':<22}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
for i, l in enumerate(labels):
    print(f"{l:<22}{p[i]:>7.3f}{r[i]:>7.3f}{f[i]:>7.3f}{s[i]:>9}")

# 2a. Exact match by number of true tactics
n_true = Y.sum(axis=1)
exact = (Y == pred).all(axis=1)
print("\nExact match by number of true tactics:")
for k in sorted(set(n_true)):
    if (n_true == k).sum() >= 5:
        print(f"  {k} tactic(s): n={(n_true == k).sum():>5}  exact match={exact[n_true == k].mean():.3f}")

# 2b. Exact match by sentence length
words = val["text_clean"].str.split().str.len()
bins = pd.cut(words, [0, 10, 20, 40, 1000], labels=["<=10", "11-20", "21-40", ">40"])
print("\nExact match by sentence length (words):")
for b in bins.cat.categories:
    m = (bins == b).values
    print(f"  {b:<6} n={m.sum():>5}  exact match={exact[m].mean():.3f}")

# 2c. Most common confusion: tactic predicted but not true (FP) / true but missed (FN)
fp = Counter(); fn = Counter(); pair = Counter()
for yt, yp in zip(Y, pred):
    miss = np.where((yt == 1) & (yp == 0))[0]
    extra = np.where((yt == 0) & (yp == 1))[0]
    for j in miss: fn[labels[j]] += 1
    for j in extra: fp[labels[j]] += 1
    for a in miss:
        for b in extra:
            pair[(labels[a], labels[b])] += 1
print("\nTop confusions (true tactic -> wrongly predicted instead):")
for (a, b), c in pair.most_common(8):
    print(f"  {a:<22}-> {b:<22}{c:>4}")

# 2d. Confident mistakes
err_idx = np.where(~exact)[0]
conf = np.array([max(P[i][(Y[i] == 0)].max(initial=0), (1 - P[i][(Y[i] == 1)]).max(initial=0)) for i in err_idx])
worst = err_idx[np.argsort(-conf)[:6]]
print(f"\nTotal sentences with at least one wrong label: {len(err_idx)} / {len(Y)} ({len(err_idx) / len(Y) * 100:.1f}%)")
print("\nMost confident mistakes:")
for i in worst:
    txt = val["text_clean"][i]
    txt = txt if len(txt) < 230 else txt[:227] + "..."
    print(f"\n  TRUE: {', '.join(labels[Y[i] == 1])}\n  PRED: {', '.join(labels[pred[i] == 1])}")
    print(textwrap.fill(txt, 88, initial_indent="  ", subsequent_indent="  "))

out = val[["text_clean", "tactics"]].copy()
out["predicted"] = ["|".join(labels[row == 1]) for row in pred]
out["correct"] = exact
out[~out["correct"]].to_csv("data/val_errors_bert.csv", index=False)
print("\nSaved: data/thresholds_bert.json, data/val_errors_bert.csv")
