"""
Step 6: Improved model - fine-tune a BERT-style transformer for multilabel tactic classification.

Designed to run on Google Colab (GPU: Runtime > Change runtime type > T4 GPU).
Trains on train.csv, selects the best epoch on val.csv. The test set is NOT touched here.

Input : data/train.csv, data/val.csv  (and data/val_predictions_baseline.csv for comparison)
Output: models/bert/ (best model), data/val_probs_bert.npy, data/val_predictions_bert.csv

Change the model with an environment variable, e.g.:
    MODEL_NAME=roberta-base python 05_finetune_bert.py
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# ---------- Config ----------
MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
THRESHOLD = 0.5           # default; tuned per tactic in the next step

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| Model:", MODEL_NAME)
if device.type == "cpu":
    print("WARNING: no GPU found. Training will be very slow. Use Colab with a GPU runtime.")

Path("models").mkdir(exist_ok=True)

# ---------- Data ----------
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
mlb = MultiLabelBinarizer()                      # same sorted label order as the baseline
Y_train = mlb.fit_transform(train["tactics"].str.split("|")).astype(np.float32)
Y_val = mlb.transform(val["tactics"].str.split("|")).astype(np.int64)
labels = list(mlb.classes_)
n_labels = len(labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(texts):
    return tokenizer(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, Y=None):
        self.ids = encode(texts)
        self.Y = Y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        item = {"input_ids": self.ids[i]}
        if self.Y is not None:
            item["labels"] = self.Y[i]
        return item


def collate(batch):
    padded = tokenizer.pad({"input_ids": [b["input_ids"] for b in batch]}, return_tensors="pt")
    if "labels" in batch[0]:
        padded["labels"] = torch.tensor(np.stack([b["labels"] for b in batch]))
    return padded


train_loader = DataLoader(TextDataset(train["text_clean"], Y_train), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)
val_loader = DataLoader(TextDataset(val["text_clean"]), batch_size=128, shuffle=False,
                        collate_fn=collate)

# ---------- Model, loss, optimizer ----------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=n_labels, problem_type="multi_label_classification").to(device)

# Class imbalance: weight positives by sqrt(neg / pos) per tactic (softer than full balancing)
pos = Y_train.sum(axis=0)
pos_weight = torch.tensor(np.sqrt((len(Y_train) - pos) / np.maximum(pos, 1)),
                          dtype=torch.float32, device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

no_decay = ["bias", "LayerNorm.weight"]
params = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(params, lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


# ---------- Metrics ----------
def force_one_label(probs, pred):
    """Every sentence has >= 1 tactic: if nothing passes the threshold, take the top label."""
    pred = pred.copy()
    empty = pred.sum(axis=1) == 0
    pred[empty, probs[empty].argmax(axis=1)] = 1
    return pred


def evaluate(Y_true, Y_pred):
    tp = (Y_true & Y_pred).sum(axis=0)
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "exact_match": float((Y_true == Y_pred).all(axis=1).mean()),
        "label_coverage": float((tp > 0).mean()),
    }


@torch.no_grad()
def predict_probs(loader):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(out)


# ---------- Training loop ----------
best_macro, best_probs, history = -1.0, None, []
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, running = time.time(), 0.0
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()

    probs = predict_probs(val_loader)
    pred = force_one_label(probs, (probs >= THRESHOLD).astype(int))
    m = evaluate(Y_val, pred)
    m.update(epoch=epoch, train_loss=running / len(train_loader), seconds=round(time.time() - t0, 1))
    history.append(m)
    print(f"Epoch {epoch}/{EPOCHS} | loss {m['train_loss']:.4f} | micro-F1 {m['micro_f1']:.4f} | "
          f"macro-F1 {m['macro_f1']:.4f} | exact {m['exact_match']:.4f} | {m['seconds']}s")

    if m["macro_f1"] > best_macro:                  # model selection uses VALIDATION only
        best_macro, best_probs, best_epoch = m["macro_f1"], probs, epoch
        model.save_pretrained("models/bert")
        tokenizer.save_pretrained("models/bert")

print(f"\nBest epoch by validation Macro-F1: {best_epoch} ({best_macro:.4f})")

# ---------- Final report on validation ----------
probs = best_probs
pred = force_one_label(probs, (probs >= THRESHOLD).astype(int))
line = "=" * 74
print(line)
print("BERT vs BASELINE ON VALIDATION SET")
print(line)
rows = {f"BERT ({MODEL_NAME})": evaluate(Y_val, pred)}
base_path = Path("data/val_predictions_baseline.csv")
if base_path.exists():
    base = pd.read_csv(base_path)
    Yb = mlb.transform(base["predicted"].str.split("|")).astype(np.int64)
    rows["Baseline (SVM word+char)"] = evaluate(Y_val, Yb)
print(pd.DataFrame(rows).T.round(4).to_string())

p, r, f, support = precision_recall_fscore_support(Y_val, pred, zero_division=0)
if base_path.exists():
    pb, rb, fb, _ = precision_recall_fscore_support(Y_val, Yb, zero_division=0)
print(f"\n{'tactic':<22}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}" + ("   baseline F1   diff" if base_path.exists() else ""))
for i, lab in enumerate(labels):
    extra = f"   {fb[i]:>11.3f} {f[i] - fb[i]:>+6.3f}" if base_path.exists() else ""
    print(f"{lab:<22}{p[i]:>7.3f}{r[i]:>7.3f}{f[i]:>7.3f}{support[i]:>9}{extra}")

mask = ~val["has_tech_name"].values
sub = evaluate(Y_val[mask], pred[mask])
print(f"\nSentences WITHOUT the technique name (n={mask.sum()}): "
      f"micro-F1 {sub['micro_f1']:.4f} | macro-F1 {sub['macro_f1']:.4f} | exact {sub['exact_match']:.4f}")

# ---------- Save ----------
np.save("data/val_probs_bert.npy", probs)
out = val[["text_clean", "tactics"]].copy()
out["predicted"] = ["|".join(np.array(labels)[row == 1]) for row in pred]
out.to_csv("data/val_predictions_bert.csv", index=False)
pd.DataFrame(history).to_csv("data/bert_training_history.csv", index=False)
print("\nSaved: models/bert/, data/val_probs_bert.npy, data/val_predictions_bert.csv, data/bert_training_history.csv")
