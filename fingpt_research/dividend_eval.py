"""Compare passage extractors on identical frozen excerpts without execution access."""
import re
import unicodedata
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from time import perf_counter

from .dividends import FIELDS, STATUSES, load_corpus

MONEY = r"(?P<currency>\b(?:KShs?|Shs|KES)\.?)\s*(?P<amount>\d+(?:\.\d+)?)"
DATE = r"(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4})"
CUES = (
    ("cancelled", r"\b(?:cancelled|withdrawn)\b"),
    ("paid", r"\b(?:was paid|already paid|successful payout)\b"),
    ("approved", r"\b(?:passed|resolved by majority|shareholders? (?:have )?approved|endorsed|has approved)\b"),
    ("proposed", r"\b(?:recommend(?:ed|s)?|propos(?:ed|ing|es)|to approve|to declare)\b"),
    ("declared", r"\b(?:declared|announce[ds]?)\b"),
)
PROMPT = (
    "Extract one dividend event from these source fragments in supplied order. They refer to "
    "the same resolution. Ignore historical comparators, total cash payouts and dividend policy. "
    "Return only supported fields, each as {value, quote}. Copy a short exact supporting quote "
    "from ONE fragment. For amount include the original currency and per-share amount in the quote. "
    "Return ISO currency (Kenyan shillings = KES) and ISO calendar dates. "
    "payment_date may be scheduled or actual; record_date is the register cutoff. "
    "Use proposed for recommendations or agenda items, approved for a passed resolution, "
    "declared for a definitive announcement, paid only for a completed payment, or cancelled. "
    "Do not infer payment from a future payment date. Omit missing fields. "
    "Fields: status, amount, currency, record_date, payment_date."
)
ITEM = {"type": "object", "properties": {"value": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["value", "quote"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {key: ITEM for key in FIELDS}, "additionalProperties": False}


def normalize(field, value):
    value = unicodedata.normalize("NFKC", value.strip())
    if field == "status":
        return value.lower() if value.lower() in STATUSES else None
    if field == "currency":
        return "KES" if value.rstrip(".").upper() in {"KES", "KSH", "KSHS", "SHS"} else None
    if field == "amount":
        try:
            amount = Decimal(value)
            if amount.is_finite() and 0 <= amount < 10**12 and amount.as_tuple().exponent >= -12:
                return format(amount.normalize(), "f")
        except InvalidOperation:
            pass
        return None
    value = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", value, flags=re.I)
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def rules(doc):
    fields = {}
    passages = doc["passages"]
    for status, pattern in CUES:
        hit = next((m for text in passages if (m := re.search(pattern, text, re.I))), None)
        if hit:
            fields["status"] = {"value": status, "quote": hit[0]}
            break
    for text in passages:
        for m in re.finditer(MONEY, text, re.I):
            if re.match(r"\s*(?:billion|million|%)", text[m.end():], re.I):
                continue
            if "amount" not in fields:
                for k in ("amount", "currency"):
                    fields[k] = {"value": normalize(k, m[k]), "quote": m[0]}
        for field, cue in (("payment_date", r"(?:paid|payable|payment|about)"), ("record_date", r"(?:close of business|register)")):
            match = re.search(cue + r"[^.;]{0,45}?(" + DATE + ")", text, re.I)
            if match and (value := normalize(field, match[1])):
                fields[field] = {"value": value, "quote": match[0]}
    return fields


def predict(backend, doc):
    raw = backend.generate_json(PROMPT, {"pages": doc["passages"]}, SCHEMA)
    output = {}
    for field in FIELDS:
        item = raw.get(field)
        if not isinstance(item, dict):
            continue
        quote, value = item.get("quote"), item.get("value")
        if not isinstance(value, str) or not isinstance(quote, str) or not quote.strip():
            continue
        if any(quote in text for text in doc["passages"]) and (value := normalize(field, value)):
            output[field] = {"value": value, "quote": quote}
    return output


def score(predictions, labels):
    by_field = {k: Counter(expected=0, returned=0, correct=0, cited=0, abstentions=0) for k in FIELDS}
    false_approvals = 0
    for key, fields in predictions.items():
        for field, counts in by_field.items():
            gold, got = labels[key].get(field), fields.get(field)
            correct = bool(gold and got and gold["value"] == got["value"])
            counts.update(expected=int(bool(gold)), returned=int(bool(got)), correct=int(correct),
                          cited=int(correct and gold["span"].casefold() in got["quote"].casefold()),
                          abstentions=int(bool(gold and not got)))
            if field == "status" and got and got["value"] == "approved" and not correct:
                false_approvals += 1
    totals = {k: sum(row[k] for row in by_field.values()) for k in next(iter(by_field.values()))}
    return {"totals": totals, "fields": by_field, "false_approvals": false_approvals,
            "predictions": predictions}


def evaluate(split, backend=None):
    docs, membership, labels, digest = load_corpus()
    docs = [d for d in docs if membership[d["id"]]["split"] == split]
    if backend and len(docs) > getattr(backend, "max_calls", len(docs)):
        raise ValueError(f"split needs {len(docs)} requests; increase --max-calls or select a smaller split")
    base = {d["id"]: rules(d) for d in docs}
    result = {"corpus_sha256": digest, "split": split, "passages": len(docs),
              "rules": score(base, labels), "reference_status": "single-reader"}
    if backend:
        model, hybrid, times = {}, {}, {}
        calls = 0
        for d in docs:
            key = d["id"]
            start = perf_counter()
            model[key] = predict(backend, d)
            times[key] = perf_counter() - start
            # Route incomplete core extractions only. Sparse optional dates alone do not trigger inference.
            if {"amount", "status", "currency"} <= base[key].keys():
                hybrid[key] = base[key]
            else:
                calls += 1
                hybrid[key] = {**model[key], **base[key]}
        result.update(model=score(model, labels), hybrid=score(hybrid, labels),
                      hybrid_routed=calls, latency_seconds=times,
                      backend={"provider": backend.provider, "model": backend.version,
                               "tokens": backend.tokens, "diagnostics": dict(backend.diagnostics)})
    return result
