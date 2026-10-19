"""
Step 2: Extract (sentence, technique, tactic) rows from MITRE ATT&CK Enterprise (STIX 2.1).

Input : data/enterprise-attack-20.0.json  (or enterprise-attack-20_0.json)
Output: data/attack_sentences.csv
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "attack_sentences.csv"

# Accept either file name: enterprise-attack-20.0.json or enterprise-attack-20_0.json
candidates = sorted(DATA_DIR.glob("enterprise-attack-*.json"))
if not candidates:
    raise FileNotFoundError(
        "Put the ATT&CK STIX file in the 'data/' folder "
        "(e.g. data/enterprise-attack-20.0.json)."
    )
RAW_PATH = candidates[-1]
print("Reading:", RAW_PATH)

bundle = json.loads(RAW_PATH.read_text(encoding="utf-8"))
objects = bundle["objects"]
print("Total objects:", len(objects))

# Record the ATT&CK version for the report
for o in objects:
    if o["type"] == "x-mitre-collection":
        print("Collection:", o.get("name"), "| version:", o.get("x_mitre_version"))
        break


# ---------- Helpers ----------
def is_active(o):
    """Skip revoked or deprecated objects."""
    return not o.get("revoked", False) and not o.get("x_mitre_deprecated", False)


def attack_id(o):
    """Return the ATT&CK ID (e.g. T1059.001) of an object."""
    for ref in o.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def clean_text(t):
    """Remove citations, markdown links and HTML tags."""
    t = re.sub(r"\(Citation:[^)]*\)", "", t)          # (Citation: ...)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)    # [text](url) -> text
    t = re.sub(r"<code>|</code>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- Technique table: id, name, tactics ----------
techniques = {}  # stix_id -> dict
for o in objects:
    if o["type"] == "attack-pattern" and is_active(o):
        tactics = [
            p["phase_name"]
            for p in o.get("kill_chain_phases", [])
            if p.get("kill_chain_name") == "mitre-attack"
        ]
        techniques[o["id"]] = {
            "technique_id": attack_id(o),
            "technique_name": o["name"],
            "tactics": tactics,
        }
print("Active techniques / sub-techniques:", len(techniques))

# ---------- Source table: groups, malware, tools, campaigns ----------
sources = {}
for o in objects:
    if o["type"] in ("intrusion-set", "malware", "tool", "campaign") and is_active(o):
        sources[o["id"]] = {"source_name": o["name"], "source_type": o["type"]}

# ---------- Extract procedure examples ----------
rows = []
for o in objects:
    if (
        o["type"] == "relationship"
        and o.get("relationship_type") == "uses"
        and is_active(o)
    ):
        src, tgt = o["source_ref"], o["target_ref"]
        if src in sources and tgt in techniques and o.get("description"):
            tech, s = techniques[tgt], sources[src]
            rows.append(
                {
                    "text": clean_text(o["description"]),
                    "technique_id": tech["technique_id"],
                    "technique_name": tech["technique_name"],
                    "tactics": "|".join(tech["tactics"]),
                    "source_stix_id": src,
                    "source_name": s["source_name"],
                    "source_type": s["source_type"],
                }
            )

df = pd.DataFrame(rows)
print("Raw procedure examples:", len(df))

# ---------- Basic filtering ----------
df = df[df["text"].str.split().str.len() >= 5]  # drop very short texts
df = df.drop_duplicates(subset=["text", "technique_id"]).reset_index(drop=True)
print("After filtering:", len(df))

df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print("Saved:", OUT_PATH)
