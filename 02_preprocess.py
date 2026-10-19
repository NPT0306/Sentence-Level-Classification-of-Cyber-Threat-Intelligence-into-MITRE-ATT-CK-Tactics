"""
Step 3: Preprocessing and leakage prevention.

Input : data/attack_sentences.csv
Output: data/clean.csv
"""
import re

import pandas as pd

df = pd.read_csv("data/attack_sentences.csv")

# Indicators of compromise (IOCs) -> placeholder tokens
IOC_PATTERNS = [
    (r"\bhttps?://\S+", " <URL> "),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " <IP> "),
    (r"\b[a-fA-F0-9]{32,64}\b", " <HASH> "),
    (r"\b(?:HKLM|HKCU|HKEY_[A-Z_]+)\\[^\s,;]+", " <REGKEY> "),
    (r"\b[A-Za-z]:\\[^\s,;]+", " <PATH> "),
    (r"\b[\w.-]+@[\w-]+\.[\w.-]+\b", " <EMAIL> "),
]


def mask_row(row):
    t = row["text"]
    # 1. Remove technique IDs such as T1059 or T1059.001
    t = re.sub(r"\bT\d{4}(?:\.\d{3})?\b", " ", t)
    # 2. Replace the group / malware name with a placeholder
    t = re.sub(re.escape(str(row["source_name"])), " <ACTOR> ", t, flags=re.I)
    # 3. Normalize IOCs
    for pat, rep in IOC_PATTERNS:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


df["text_clean"] = df.apply(mask_row, axis=1)

# Flag (do not delete) sentences that contain the technique name verbatim.
# Used later for an ablation study.
df["has_tech_name"] = df.apply(
    lambda r: str(r["technique_name"]).lower() in r["text_clean"].lower(), axis=1
)

df = df[df["text_clean"].str.split().str.len() >= 5]


# Merge duplicate sentences: union of tactic / technique labels
def union(series, sep="|"):
    return sep.join(sorted({x for s in series for x in str(s).split(sep)}))


g = df.groupby("text_clean", as_index=False).agg(
    tactics=("tactics", union),
    technique_ids=("technique_id", lambda s: "|".join(sorted(set(s)))),
    sources=("source_stix_id", lambda s: "|".join(sorted(set(s)))),
    has_tech_name=("has_tech_name", "max"),
)
g["n_sources"] = g["sources"].str.count(r"\|") + 1
# Temporary grouping key (Step 4 will handle grouping properly)
g["group"] = g["sources"].str.split("|").str[0]

# ---------- Clean summary ----------
import textwrap

def pct(x):
    return f"{x * 100:.1f}%"

line = "=" * 60
print(line)
print("PREPROCESSING SUMMARY")
print(line)
print(f"Unique sentences              : {len(g):,}")
print(f"With more than 1 tactic       : {pct(g['tactics'].str.contains(r'[|]').mean())}")
print(f"With more than 1 technique    : {pct(g['technique_ids'].str.contains(r'[|]').mean())}")
print(f"Containing technique name     : {pct(g['has_tech_name'].mean())}")
print(f"Coming from more than 1 source: {pct((g['n_sources'] > 1).mean())}")

print("\nTactics per sentence:")
for k, v in g["tactics"].str.count(r"[|]").add(1).value_counts().sort_index().items():
    print(f"  {k} tactic(s): {v:,}")

print("\nRandom samples:")
for i, (_, r) in enumerate(g.sample(5, random_state=3).iterrows(), 1):
    text = r["text_clean"]
    if len(text) > 200:
        text = text[:197] + "..."
    print(f"\n[{i}] tactics: {r['tactics'].replace('|', ', ')}")
    print(textwrap.fill(text, width=90, initial_indent="    ", subsequent_indent="    "))
print()

g.to_csv("data/clean.csv", index=False)
print("Saved: data/clean.csv")
