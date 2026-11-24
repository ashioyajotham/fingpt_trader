"""Export blind annotations or reconcile a returned packet against the frozen references."""
import argparse
import json
from pathlib import Path

from asof.dividends import CORPUS, FIELDS, load_corpus


def packet():
    notices, _, _, digest = load_corpus()
    sources = json.loads((CORPUS / "locators.json").read_text())
    return {
        "corpus_sha256": digest,
        "instructions": [
            "Read each numbered set of fragments with its source. The target is the event described in those fragments.",
            "For each supported field enter a normalized value and the shortest exact supporting span copied from one fragment. Preserve case and punctuation.",
            "Use proposed, declared, approved, paid or cancelled. A proposed resolution needs a passed vote or other explicit decision to count as approved.",
            "Use a decimal per-share amount, ISO currency and YYYY-MM-DD dates. A payout percentage or par value is a different quantity. A future payment date may be recorded.",
            "Leave both entries empty for absent fields. Use notes for ambiguous amounts, dividend scope, date precision, inaccurate excerpts or sources.",
            "Set reviewed to true after checking each passage. Save this file and return it for reconciliation."
        ],
        "reviewer": "",
        "notices": [{**doc, "locator": sources.get(doc["id"], {}), "reviewed": False, "notes": "",
                     "fields": {field: {"value": "", "span": ""} for field in FIELDS}} for doc in notices],
    }


def reconcile(review):
    docs, _, refs, digest = load_corpus()
    rows = review["notices"]
    if not review.get("reviewer", "").strip() or review["corpus_sha256"] != digest:
        raise ValueError("supply a reviewer name and the matching corpus hash")
    keys = [r["id"] for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(refs):
        raise ValueError("expected each frozen passage exactly once")
    originals = {d["id"]: d for d in docs}
    differences, open_rows = [], []
    for row in rows:
        key = row["id"]
        if row.get("reviewed") is not True:
            open_rows.append(key)
            continue
        if set(row["fields"]) != set(FIELDS):
            raise ValueError(f"missing or unknown field in {key}")
        for field, answer in row["fields"].items():
            value, span = answer["value"].strip(), answer["span"]
            if bool(value) != bool(span) or (span and not any(span in p for p in originals[key]["passages"])):
                raise ValueError(f"expected a verbatim span and value for {key} {field}")
            gold = refs[key].get(field)
            if (gold or {}).get("value", "") != value or (gold or {}).get("span", "") != span:
                differences.append({"id": key, "field": field, "reference": gold, "reader": answer})
    return {"reviewer": review["reviewer"], "unreviewed": open_rows, "differences": differences,
            "notes": {r["id"]: r["notes"] for r in rows if r.get("notes")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned", type=Path)
    args = parser.parse_args()
    result = reconcile(json.loads(args.returned.read_text())) if args.returned else packet()
    print(json.dumps(result, indent=2, ensure_ascii=False))
