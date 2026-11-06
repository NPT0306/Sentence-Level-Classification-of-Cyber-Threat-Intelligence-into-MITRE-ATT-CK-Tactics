# Classifying Threat Intelligence Sentences into MITRE ATT&CK Tactics

AIC211 – AI for Cybersecurity · Project Topic 10: Threat Intelligence / ATT&CK Technique Classification

**Team:** Đoàn Hải Đăng, Chu Văn Tuấn Đạt, Lê Đức Cường, Pruvot Chales, Nguyễn Phúc Thịnh

## 1. What this project does

The model reads **one sentence** from a cyber threat intelligence (CTI) description and predicts which of the
**15 MITRE ATT&CK Enterprise tactics (v20.0)** it expresses. This is a multilabel text classification task:
one sentence can carry several tactics (13.5% of the sentences do).

Two models are compared:

| Model | Description |
|---|---|
| Baseline | TF-IDF (word 1–2 grams + character 3–5 grams) with a one-vs-rest Linear SVM |
| Improved model | Fine-tuned `distilbert-base-uncased` with 15 sigmoid outputs and a weighted loss |

Data are split **by source group** (malware / group / campaign) so that near-identical sentences never appear in
both training and test data.

### Main results (test set, 2,600 sentences, evaluated once)

| Model | Micro-F1 | Macro-F1 | Exact match |
|---|---|---|---|
| Baseline (Linear SVM) | 0.805 | 0.763 | 0.710 |
| DistilBERT | 0.835 | 0.791 | 0.781 |

The Macro-F1 gain is +0.028 with a 95% bootstrap interval of [+0.010, +0.046].

## 2. Folder structure

```
.
├── README.md                       this file
├── requirements.txt                Python dependencies
├── 01_build_dataset.py             extract (sentence, technique, tactic) rows from ATT&CK STIX
├── 02_preprocess.py                clean text, mask actors and indicators, merge duplicates
├── 03_split_data.py                group-aware, multilabel-stratified 70/15/15 split
├── 04_baseline.py                  TF-IDF + Logistic Regression / Linear SVM (validation only)
├── 05_finetune_bert.py             fine-tune DistilBERT (GPU recommended, e.g. Google Colab T4)
├── 06_threshold_and_errors.py      honest threshold tuning and validation error analysis
├── 07_final_test.py                final evaluation on the test set (run once)
├── 08_explain_demo.py              occlusion explanations, robustness test, web demo
├── AIC211_Project_Topic10_ATTACK_Tactic_Pipeline.ipynb   the whole pipeline in one notebook
├── fig1_tactic_distribution.png    figures produced by the notebook
├── fig2_model_comparison.png
├── fig3_per_tactic_f1.png
├── fig4_tactic_confusion.png
├── data/                           datasets and intermediate results (see section 4)
└── models/                         saved models (see section 5)
```

## 3. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Download the ATT&CK Enterprise v20.0 STIX file from
<https://github.com/mitre-attack/attack-stix-data> (folder `enterprise-attack/`) and save it as
`data/enterprise-attack-20.0.json` (the name `enterprise-attack-20_0.json` also works).

## 4. The `data/` directory

Input (you provide it):

| File | Description |
|---|---|
| `enterprise-attack-20.0.json` | MITRE ATT&CK Enterprise v20.0 in STIX 2.1 format (public data by MITRE, used with attribution) |

Files created by the scripts, in the order they are produced:

| File | Created by | Description |
|---|---|---|
| `attack_sentences.csv` | 01 | 17,563 procedure examples: text, technique, tactics, source |
| `clean.csv` | 02 | 16,952 unique cleaned sentences with columns `text_clean`, `tactics`, `technique_ids`, `sources`, `has_tech_name`, `group` |
| `train.csv`, `val.csv`, `test.csv` | 03 | 11,752 / 2,600 / 2,600 sentences; tactics are joined with `\|`; `group_id` is the split group |
| `val_predictions_baseline.csv` | 04 | baseline predictions on validation data |
| `val_probs_bert.npy`, `val_predictions_bert.csv`, `bert_training_history.csv` | 05 | DistilBERT probabilities and predictions on validation data, training log |
| `thresholds_bert.json`, `val_errors_bert.csv` | 06 | decision threshold per tactic (0.5 for all), validation errors |
| `test_predictions.csv`, `test_results.csv` | 07 | test predictions of both models and the final metrics |
| `explanations.html`, `explanations.csv`, `robustness_results.csv` | 08 | word-importance examples and the noise-robustness table |

Label columns hold tactic names such as `credential-access`, joined with `|` when a sentence has several tactics.

## 5. The `models/` directory

| Path | Description |
|---|---|
| `models/baseline.joblib` | best baseline (vectorizers and classifier), written by `04_baseline.py` |
| `models/bert/` | fine-tuned DistilBERT: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, written by `05_finetune_bert.py` |

## 6. How to run

### Option A: scripts, in order (seed 42)

```bash
python 01_build_dataset.py          # needs data/enterprise-attack-20.0.json
python 02_preprocess.py
python 03_split_data.py
python 04_baseline.py
python 05_finetune_bert.py          # GPU recommended; on Google Colab choose a T4 GPU runtime
python 06_threshold_and_errors.py
python 07_final_test.py             # evaluates on the test set: run once, after all choices are frozen
```

Scripts 05, 07 and 08 need `torch` and `transformers`. If you have no GPU, run script 05 on Google Colab and copy
`models/bert/` and the `data/*bert*` files back into this folder.

### Option B: the notebook

Open `AIC211_Project_Topic10_ATTACK_Tactic_Pipeline.ipynb` and run all cells from top to bottom. It repeats the whole
pipeline (steps 0–13). Settings are at the top of the first cell:

| Setting | Meaning |
|---|---|
| `RUN_BERT` | run the DistilBERT step |
| `FORCE_RETRAIN_BERT` | `False`: reuse `models/bert/` if it exists |
| `USE_SAVED_BERT_OUTPUTS` | no `torch`? reuse the saved DistilBERT outputs in `data/` |
| `RUN_DEMO` | launch the web demo at the end |

### Explanations, robustness and web demo

```bash
python 08_explain_demo.py examples      # word importance for example sentences -> data/explanations.html
python 08_explain_demo.py robustness    # stability under synonyms, typos, extra sentence
python 08_explain_demo.py demo          # web demo (needs gradio): paste a report, see tactics and key words
```

## 7. Notes and limitations

- The test set is used once, in script 07 (notebook step 8). Do not tune anything after looking at it.
- Labels belong to ATT&CK techniques, not to individual sentences, so some sentence labels are noisy.
- All sentences come from ATT&CK itself; real threat reports have not been tested.
- The classifier always returns at least one tactic, even for unrelated text.
- The model is decision support for analysts and its output needs human review.
- Tested with Python 3.11, scikit-learn 1.9, pandas 3.0 and numpy 2.5; the DistilBERT run used `transformers` 5.20.

## 8. Data, tools and AI-assistance declaration

- **Data:** MITRE ATT&CK Enterprise v20.0 (STIX 2.1), public data by MITRE.
- **Pretrained model:** `distilbert-base-uncased` from Hugging Face, fine-tuned by the team.
- **Libraries:** pandas, NumPy, SciPy, scikit-learn, PyTorch, Hugging Face Transformers, Matplotlib, Gradio.
- **Generative AI:** a generative AI assistant (Claude, by Anthropic) was used while developing this project.
  [The team must state here which parts it designed, implemented, ran and checked itself.]
