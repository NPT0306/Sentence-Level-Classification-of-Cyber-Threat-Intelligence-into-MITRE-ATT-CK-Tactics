"""
Step 7b: FINAL evaluation on the test set. Run this ONCE, after every choice is frozen.

Frozen choices:
  - Baseline : SVM, word 1-2gram + char 3-5gram TF-IDF (trained on train.csv)
  - Improved : fine-tuned DistilBERT in models/bert/ (best epoch picked on val)
  - Threshold: values in data/thresholds_bert.json (0.5 for every tactic)

Needs: data/train.csv, data/test.csv, data/thresholds_bert.json, models/bert/
       (models/bert/ must contain config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
Output: data/test_predictions.csv, data/test_results.csv
Works on CPU (about a few minutes) or GPU.
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train["tactics"].str.split("|"))
Y_test = mlb.transform(test["tactics"].str.split("|"))
labels = np.array(mlb.classes_)
thr = np.array([json.load(open("data/thresholds_bert.json"))[l] for l in labels])


def force_one_label(scores, pred):
    pred = pred.copy()
    empty = pred.sum(axis=1) == 0
    pred[empty, scores[empty].argmax(axis=1)] = 1
    return pred


def evaluate(Y, pred):
    tp = (Y & pred).sum(axis=0)
    return {"micro_f1": f1_score(Y, pred, average="micro", zero_division=0),
            "macro_f1": f1_score(Y, pred, average="macro", zero_division=0),
            "exact_match": float((Y == pred).all(axis=1).mean()),
            "label_coverage": float((tp > 0).mean())}


# ---------- Baseline (retrained on train only, same config as Step 5) ----------
print("Training baseline...")
word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150_000)
Xtr = hstack([word.fit_transform(train["text_clean"]), char.fit_transform(train["text_clean"])]).tocsr()
Xte = hstack([word.transform(test["text_clean"]), char.transform(test["text_clean"])]).tocsr()
svm = OneVsRestClassifier(LinearSVC(C=0.5, class_weight="balanced"), n_jobs=-1).fit(Xtr, Y_train)
scores = svm.decision_function(Xte)
pred_svm = force_one_label(scores, (scores > 0).astype(int))

# ---------- BERT ----------
print("Running BERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("models/bert")
model = AutoModelForSequenceClassification.from_pretrained("models/bert").to(device).eval()
texts, probs = list(test["text_clean"]), []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        probs.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
P = np.vstack(probs)
pred_bert = (P >= thr).astype(int)
empty = pred_bert.sum(axis=1) == 0
pred_bert[empty, (P[empty] / thr).argmax(axis=1)] = 1

# ---------- Results ----------
line = "=" * 70
print("\n" + line); print("FINAL TEST RESULTS"); print(line)
res = pd.DataFrame({"Baseline (SVM word+char)": evaluate(Y_test, pred_svm),
                    "DistilBERT": evaluate(Y_test, pred_bert)}).T
print(res.round(4).to_string())

p1, r1, f1_, s = precision_recall_fscore_support(Y_test, pred_svm, zero_division=0)
p2, r2, f2_, _ = precision_recall_fscore_support(Y_test, pred_bert, zero_division=0)
print(f"\n{'tactic':<22}{'support':>8}{'SVM F1':>9}{'BERT F1':>9}{'diff':>8}{'BERT P':>8}{'BERT R':>8}")
for i, l in enumerate(labels):
    print(f"{l:<22}{s[i]:>8}{f1_[i]:>9.3f}{f2_[i]:>9.3f}{f2_[i] - f1_[i]:>+8.3f}{p2[i]:>8.3f}{r2[i]:>8.3f}")

# Sentences that do not contain the technique name (fairer test of semantics)
m = ~test["has_tech_name"].values
print(f"\nOnly sentences WITHOUT the technique name (n={m.sum()}):")
for name, pr in (("SVM", pred_svm), ("BERT", pred_bert)):
    e = evaluate(Y_test[m], pr[m])
    print(f"  {name:<5} micro-F1 {e['micro_f1']:.4f} | macro-F1 {e['macro_f1']:.4f} | exact {e['exact_match']:.4f}")

# Bootstrap 95% confidence intervals (resampling test sentences)
rng = np.random.default_rng(42)
diffs, mb, ms = [], [], []
for _ in range(1000):
    idx = rng.integers(0, len(Y_test), len(Y_test))
    a = f1_score(Y_test[idx], pred_bert[idx], average="macro", zero_division=0)
    b = f1_score(Y_test[idx], pred_svm[idx], average="macro", zero_division=0)
    mb.append(a); ms.append(b); diffs.append(a - b)
ci = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
print(f"\nBootstrap 95% CI for Macro-F1:  BERT {ci(mb)} | SVM {ci(ms)}")
print(f"Macro-F1 difference (BERT - SVM): {np.mean(diffs):+.3f}  95% CI {ci(diffs)}")

out = test[["text_clean", "tactics"]].copy()
out["pred_svm"] = ["|".join(labels[r == 1]) for r in pred_svm]
out["pred_bert"] = ["|".join(labels[r == 1]) for r in pred_bert]
out.to_csv("data/test_predictions.csv", index=False)
res.to_csv("data/test_results.csv")
print("\nSaved: data/test_predictions.csv, data/test_results.csv")
