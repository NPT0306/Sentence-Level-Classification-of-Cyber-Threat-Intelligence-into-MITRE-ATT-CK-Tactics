"""
Step 5: Baseline models (TF-IDF + linear classifiers, one-vs-rest).

Trains on train.csv and evaluates on val.csv ONLY. The test set is not touched here.

Input : data/train.csv, data/val.csv
Output: data/val_predictions_baseline.csv, models/baseline.joblib
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

Path("models").mkdir(exist_ok=True)
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train["tactics"].str.split("|"))
Y_val = mlb.transform(val["tactics"].str.split("|"))
labels = list(mlb.classes_)


# ---------- Metrics ----------
def evaluate(Y_true, Y_pred):
    tp = (Y_true & Y_pred).sum(axis=0)
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "exact_match": float((Y_true == Y_pred).all(axis=1).mean()),
        "label_coverage": float((tp > 0).mean()),   # labels predicted correctly at least once
    }


def force_one_label(scores, Y_pred):
    """Every sentence has >= 1 tactic: if nothing is predicted, take the top-scoring label."""
    Y_pred = Y_pred.copy()
    empty = Y_pred.sum(axis=1) == 0
    Y_pred[empty, scores[empty].argmax(axis=1)] = 1
    return Y_pred


# ---------- Features ----------
def build_features(use_char):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    parts = [word]
    if use_char:
        parts.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True, max_features=150_000))
    return parts


def fit_features(parts, texts_train):
    return hstack([p.fit_transform(texts_train) for p in parts]).tocsr()


def transform(parts, texts):
    return hstack([p.transform(texts) for p in parts]).tocsr()


# ---------- Experiments ----------
experiments = {
    "LR  | word 1-2gram":        (False, lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
    "SVM | word 1-2gram":        (False, lambda: LinearSVC(C=0.5, class_weight="balanced")),
    "LR  | word + char n-gram":  (True,  lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
    "SVM | word + char n-gram":  (True,  lambda: LinearSVC(C=0.5, class_weight="balanced")),
}

results, store = [], {}
for name, (use_char, make_clf) in experiments.items():
    t0 = time.time()
    parts = build_features(use_char)
    Xtr = fit_features(parts, train["text_clean"])
    Xva = transform(parts, val["text_clean"])
    clf = OneVsRestClassifier(make_clf(), n_jobs=-1).fit(Xtr, Y_train)
    scores = clf.decision_function(Xva)
    pred = force_one_label(scores, (scores > 0).astype(int))
    m = evaluate(Y_val, pred)
    m["model"] = name
    m["seconds"] = round(time.time() - t0, 1)
    results.append(m)
    store[name] = (parts, clf, pred)

res = pd.DataFrame(results).set_index("model")[["micro_f1", "macro_f1", "exact_match", "label_coverage", "seconds"]]
line = "=" * 74
print(line)
print("BASELINE RESULTS ON VALIDATION SET")
print(line)
print(res.round(4).to_string())

# ---------- Best baseline: per-class report ----------
best = res["macro_f1"].idxmax()
print(f"\nBest baseline by Macro-F1: {best}")
parts, clf, pred = store[best]
p, r, f, support = precision_recall_fscore_support(Y_val, pred, zero_division=0)
print(f"\n{'tactic':<22}{'precision':>10}{'recall':>9}{'F1':>8}{'support':>9}")
for i, lab in enumerate(labels):
    print(f"{lab:<22}{p[i]:>10.3f}{r[i]:>9.3f}{f[i]:>8.3f}{support[i]:>9}")

# ---------- Easy vs hard sentences (technique name leakage check) ----------
mask = ~val["has_tech_name"].values
print("\nSentences WITHOUT the technique name in text (harder, fairer test of semantics):")
sub = evaluate(Y_val[mask], pred[mask])
print(f"  n={mask.sum()}  micro_f1={sub['micro_f1']:.4f}  macro_f1={sub['macro_f1']:.4f}  exact_match={sub['exact_match']:.4f}")
sub = evaluate(Y_val[~mask], pred[~mask])
print("Sentences WITH the technique name in text:")
print(f"  n={(~mask).sum()}  micro_f1={sub['micro_f1']:.4f}  macro_f1={sub['macro_f1']:.4f}  exact_match={sub['exact_match']:.4f}")

# ---------- Top words per tactic (explainability preview, from the word-level LR model) ----------
lr_parts, lr_clf, _ = store["LR  | word 1-2gram"]
names = np.array(lr_parts[0].get_feature_names_out())
print("\nTop 6 words/phrases per tactic (Logistic Regression, largest positive weights):")
for i, lab in enumerate(labels):
    top = names[np.argsort(lr_clf.estimators_[i].coef_[0])[-6:][::-1]]
    print(f"  {lab:<22}{', '.join(top)}")

# ---------- Save ----------
out = val[["text_clean", "tactics"]].copy()
out["predicted"] = ["|".join(np.array(labels)[row == 1]) for row in pred]
out.to_csv("data/val_predictions_baseline.csv", index=False)
joblib.dump({"parts": parts, "clf": clf, "labels": labels, "name": best}, "models/baseline.joblib")
print("\nSaved: data/val_predictions_baseline.csv, models/baseline.joblib")
