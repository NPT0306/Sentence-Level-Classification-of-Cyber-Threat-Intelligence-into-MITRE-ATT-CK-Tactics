"""
Step 8: Explainability, robustness check and a small demo for the DistilBERT tactic classifier.

Modes:
    python 08_explain_demo.py examples     # word-importance explanations for example sentences
                                           # -> data/explanations.html, data/explanations.csv
    python 08_explain_demo.py robustness   # how stable are predictions under rewording / typos
                                           # -> data/robustness_results.csv
    python 08_explain_demo.py demo         # interactive web demo (needs: pip install gradio)

Needs: models/bert/ (config.json, model.safetensors, tokenizer files) and, for 'robustness', data/test.csv.
Works on CPU or GPU. No model is retrained or changed here; this step only analyzes the frozen model.

Explanation method: OCCLUSION. For a sentence, remove one word at a time, run the model again and measure
how much the probability of the target tactic drops. A large drop = that word supports the prediction.
"""
import argparse
import html
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Same (alphabetical) label order that was used for training.
LABELS = ["collection", "command-and-control", "credential-access", "defense-impairment", "discovery",
          "execution", "exfiltration", "impact", "initial-access", "lateral-movement", "persistence",
          "privilege-escalation", "reconnaissance", "resource-development", "stealth"]
THRESHOLD = 0.5
MODEL_DIR = "models/bert"

EXAMPLES = [
    "<ACTOR> used ntdsutil to obtain a copy of the victim environment ntds.dit file.",
    "<ACTOR> has used scheduled tasks to establish persistence for installed tools.",
    "<ACTOR> enables SMBv1 during execution.",
    "<ACTOR> has attempted to collect browser information.",
    "<ACTOR> can embed a copy of PsExec within its payload and place it in a temporary directory.",
    "The attackers sent spearphishing emails with a malicious attachment, then used PowerShell to download "
    "a second-stage payload and added a registry run key so it would start at every login.",
]


# ---------------------------------------------------------------- model
def load_bert(model_dir=MODEL_DIR):
    """Return probs_fn(list_of_texts) -> array [n, 15] of tactic probabilities."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    @torch.no_grad()
    def probs_fn(texts, batch_size=64):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=128,
                      padding=True, return_tensors="pt").to(device)
            out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
        return np.vstack(out)

    return probs_fn


def predict_labels(p, threshold=THRESHOLD):
    """Indices of predicted tactics; at least one (the top one) is always returned."""
    idx = np.where(p >= threshold)[0]
    return idx if len(idx) else np.array([int(np.argmax(p))])


# ---------------------------------------------------------------- explanation
def explain(sentence, probs_fn, tactic=None):
    """Word-level occlusion explanation for one sentence.

    tactic=None -> explain the most probable predicted tactic.
    Returns dict: words, probs, predicted, target, scores (probability drop when the word is removed).
    """
    words = sentence.split()
    variants = [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    P = probs_fn([sentence] + variants)
    p_full, p_var = P[0], P[1:]
    predicted = predict_labels(p_full)
    target = LABELS.index(tactic) if tactic else int(predicted[np.argmax(p_full[predicted])])
    return {"words": words, "probs": p_full, "predicted": predicted, "target": target,
            "scores": p_full[target] - p_var[:, target]}


def top_words(ex, k=4):
    order = np.argsort(-ex["scores"])[:k]
    return [(ex["words"][i], float(ex["scores"][i])) for i in order if ex["scores"][i] > 0.01]


def to_html(sentence, ex):
    """Words shaded in red in proportion to how much they support the target tactic."""
    scores = ex["scores"]
    mx = max(float(scores.max()), 1e-6)
    spans = []
    for w, s in zip(ex["words"], scores):
        a = max(0.0, float(s)) / mx
        spans.append(f"<span style='background:rgba(220,50,50,{a:.2f});padding:1px 3px;border-radius:3px' "
                     f"title='{s:+.3f}'>{html.escape(w)}</span>")
    tags = ", ".join(f"{LABELS[i]} ({ex['probs'][i]:.2f})" for i in ex["predicted"])
    return (f"<div style='font-family:sans-serif;margin:14px 0'><div style='line-height:2'>{' '.join(spans)}</div>"
            f"<div style='color:#555;font-size:13px'>Predicted: <b>{tags}</b> | shading explains "
            f"<b>{LABELS[ex['target']]}</b></div></div>")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def run_examples(probs_fn):
    rows, blocks = [], []
    for s in EXAMPLES:
        ex = explain(s, probs_fn)
        pred = ", ".join(f"{LABELS[i]} ({ex['probs'][i]:.2f})" for i in ex["predicted"])
        tw = ", ".join(f"{w} ({v:+.2f})" for w, v in top_words(ex))
        print(f"\n{s}\n  predicted : {pred}\n  key words : {tw}   [explaining {LABELS[ex['target']]}]")
        rows.append({"sentence": s, "predicted": pred, "explained_tactic": LABELS[ex["target"]], "key_words": tw})
        blocks.append(to_html(s, ex))
    Path("data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("data/explanations.csv", index=False)
    Path("data/explanations.html").write_text(
        "<html><meta charset='utf-8'><body style='max-width:900px;margin:30px auto'>"
        "<h3>Word importance (occlusion) for DistilBERT predictions</h3>" + "".join(blocks) + "</body></html>",
        encoding="utf-8")
    print("\nSaved: data/explanations.html (open in a browser, good for screenshots), data/explanations.csv")


# ---------------------------------------------------------------- robustness
SYNONYMS = {"used": "leveraged", "uses": "leverages", "use": "leverage", "downloads": "retrieves",
            "download": "retrieve", "executes": "runs", "executed": "ran", "creates": "establishes",
            "created": "established", "obtain": "acquire", "obtained": "acquired", "data": "information",
            "files": "documents", "file": "document", "collect": "gather", "collected": "gathered",
            "sends": "transmits", "sent": "transmitted", "steal": "harvest", "stole": "harvested"}


def synonym_perturb(s, rng):
    out = []
    for w in s.split():
        core = w.strip(".,;:()")
        rep = SYNONYMS.get(core.lower())
        out.append(w.replace(core, rep) if rep else w)
    return " ".join(out)


def typo_perturb(s, rng, p=0.15):
    out = []
    for w in s.split():
        if len(w) > 4 and not w.startswith("<") and rng.random() < p:
            i = rng.randrange(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def distractor_perturb(s, rng):
    return s + " The report was published in 2020."


PERTURBATIONS = {"synonym replacement": synonym_perturb, "random typos": typo_perturb,
                 "benign sentence appended": distractor_perturb}


def run_robustness(probs_fn, n=500, seed=42):
    test = pd.read_csv("data/test.csv").sample(n=n, random_state=seed).reset_index(drop=True)
    Y = np.zeros((len(test), len(LABELS)), dtype=int)
    for i, t in enumerate(test["tactics"]):
        for lab in t.split("|"):
            Y[i, LABELS.index(lab)] = 1

    def to_sets(P):
        out = np.zeros_like(P, dtype=int)
        for i, p in enumerate(P):
            out[i, predict_labels(p)] = 1
        return out

    from sklearn.metrics import f1_score
    base = to_sets(probs_fn(test["text_clean"]))
    rows = [{"input": "original", "label sets unchanged": 1.0,
             "micro-F1": f1_score(Y, base, average="micro", zero_division=0),
             "macro-F1": f1_score(Y, base, average="macro", zero_division=0)}]
    rng = random.Random(seed)
    for name, fn in PERTURBATIONS.items():
        pert = to_sets(probs_fn([fn(s, rng) for s in test["text_clean"]]))
        rows.append({"input": name, "label sets unchanged": float((pert == base).all(axis=1).mean()),
                     "micro-F1": f1_score(Y, pert, average="micro", zero_division=0),
                     "macro-F1": f1_score(Y, pert, average="macro", zero_division=0)})
    res = pd.DataFrame(rows)
    print(f"\nRobustness on {n} random test sentences (analysis only, model unchanged):")
    print(res.round(3).to_string(index=False))
    res.to_csv("data/robustness_results.csv", index=False)
    print("\nSaved: data/robustness_results.csv")


# ---------------------------------------------------------------- demo
def run_demo():
    import gradio as gr
    probs_fn = load_bert()

    def fn(text, sent_no, tactic):
        sents = split_sentences(text)
        if not sents:
            return pd.DataFrame(), [], "Please enter some text."
        P = probs_fn(sents)
        rows = [{"#": i + 1, "sentence": s,
                 "predicted tactics": ", ".join(f"{LABELS[j]} ({p[j]:.2f})" for j in predict_labels(p))}
                for i, (s, p) in enumerate(zip(sents, P))]
        k = min(max(int(sent_no or 1), 1), len(sents)) - 1
        ex = explain(sents[k], probs_fn, None if tactic == "auto" else tactic)
        shown = [(w + " ", float(sc)) for w, sc in zip(ex["words"], ex["scores"])]
        note = (f"Explaining sentence {k + 1} for tactic **{LABELS[ex['target']]}** "
                f"(probability {ex['probs'][ex['target']]:.2f}). Positive scores: removing the word lowers "
                f"the probability, so the word supports this tactic.")
        return pd.DataFrame(rows), shown, note

    demo = gr.Interface(
        fn=fn,
        inputs=[gr.Textbox(lines=6, label="Paste a threat report paragraph (one or more sentences)"),
                gr.Number(value=1, precision=0, label="Sentence number to explain"),
                gr.Dropdown(["auto"] + LABELS, value="auto", label="Tactic to explain")],
        outputs=[gr.Dataframe(label="Predicted ATT&CK tactics per sentence"),
                 gr.HighlightedText(label="Word importance (occlusion)"),
                 gr.Markdown()],
        title="ATT&CK tactic classifier (DistilBERT)",
        description="Decision support for analysts: predictions can be wrong, please review them.",
        examples=[[EXAMPLES[-1], 1, "auto"], [EXAMPLES[0], 1, "auto"]])
    demo.launch(share=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="examples", choices=["examples", "robustness", "demo"])
    args = ap.parse_args()
    if args.mode == "demo":
        run_demo()
    else:
        pf = load_bert()
        run_examples(pf) if args.mode == "examples" else run_robustness(pf)
