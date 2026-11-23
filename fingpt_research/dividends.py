"""Source-linked dividend observations and historical disclosure queries."""
import hashlib
import json
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .data import FIXTURES, instant

CORPUS = FIXTURES / "dividend-notices"
NAIROBI = ZoneInfo("Africa/Nairobi")
FIELDS = ("status", "amount", "currency", "record_date", "payment_date")
STATUSES = ("proposed", "declared", "approved", "paid", "cancelled")
SNAPSHOT = "2026-11-23T15:55:00Z"


def load_corpus(root=CORPUS):
    parts = {}
    digest = hashlib.sha256()
    for name in ("notices", "membership", "reference"):
        raw = (root / (name + ".json")).read_bytes()
        digest.update(name.encode() + b"\0" + raw)
        parts[name] = json.loads(raw)
    notices, membership, reference = (parts[k] for k in ("notices", "membership", "reference"))
    ids = {row["id"] for row in notices}
    if len(ids) != len(notices) or ids != set(reference) or ids != set(membership):
        raise ValueError("notice, membership and reference IDs must match uniquely")
    events = {}
    for row in notices:
        key = row["id"]
        group = membership[key]
        if group["split"] not in {"development", "holdout"}:
            raise ValueError("unknown evaluation split")
        if group["event"] and events.setdefault(group["event"], group["split"]) != group["split"]:
            raise ValueError("dividend event appears on both sides of the split")
        for field, item in reference[key].items():
            if field not in FIELDS or not item["span"] or not any(
                item["span"] in passage for passage in row["passages"]
            ):
                raise ValueError(f"unsupported reference {key} {field}")
    return notices, membership, reference, digest.hexdigest()


def cutoff(value):
    if len(value) == 10:
        return datetime.combine(date.fromisoformat(value), time.max, NAIROBI)
    return instant(value)


def actions(as_of, symbol=None, status=None, history=False, clock="disclosure", root=CORPUS):
    docs, groups, labels, digest = load_corpus(root)
    until = cutoff(as_of)
    state, changes = {}, []
    for doc in sorted(docs, key=lambda d: (d["notice_date"], d["id"])):
        key = doc["id"]
        event = groups[key]["event"]
        next_day = date.fromisoformat(doc["notice_date"]) + timedelta(days=1)
        available = datetime.combine(next_day, time.min, NAIROBI)
        if clock == "observed":
            available = max(instant(SNAPSHOT), available)
        if available > until or not event or (symbol and doc["symbol"] != symbol.upper()):
            continue
        prior = state.get(event, {})
        fields = prior.get("fields", {}).copy()
        for field, item in labels[key].items():
            fields[field] = {**item, "passage_id": key, "location": doc["location"],
                             "source": doc["source"], "notice_date": doc["notice_date"]}
        entry = {"event": event, "symbol": doc["symbol"], "available_at": available.isoformat(),
                 "notice_date": doc["notice_date"], "passage_id": key, "fields": fields,
                 "passages": doc["passages"], "source": doc["source"], "location": doc["location"]}
        state[event] = entry
        changes.append(entry)
    rows = changes if history else list(state.values())
    return {"clock": clock, "as_of": until.isoformat(), "reference_status": "single-reader",
            "corpus_sha256": digest, "rows": [r for r in rows if not status or
                                              r["fields"].get("status", {}).get("value") == status]}
