"""Source-linked dividend observations and historical disclosure queries."""
import hashlib
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import FIXTURES, instant

CORPUS = FIXTURES / "dividend-notices"
NAIROBI = ZoneInfo("Africa/Nairobi")
FIELDS = ("status", "amount", "currency", "record_date", "payment_date")
STATUSES = ("proposed", "declared", "approved", "paid", "cancelled")
SNAPSHOT = "2026-11-23T16:05:00Z"


def load_corpus(root=CORPUS, *, reviewed=False):
    root = Path(root)
    parts = {}
    digest = hashlib.sha256()
    for name in ("notices", "membership", "reference"):
        raw = (root / (name + ".json")).read_bytes()
        digest.update(name.encode() + b"\0" + raw)
        parts[name] = json.loads(raw)
    notices, membership, reference = (parts[k] for k in ("notices", "membership", "reference"))
    adjudication_path = root / "adjudication.json"
    if reviewed and adjudication_path.exists():
        raw = adjudication_path.read_bytes()
        review = json.loads(raw)
        if review["original_corpus_sha256"] != digest.hexdigest():
            raise ValueError("review belongs to a different frozen corpus")
        for key, updates in review["accepted_fields"].items():
            reference[key].update(updates)
        digest.update(b"adjudication\0" + raw)
    ids = {row["id"] for row in notices}
    if len(ids) != len(notices) or ids != set(reference) or ids != set(membership):
        raise ValueError("notice, membership and reference IDs must match uniquely")
    events = {}
    for row in notices:
        key = row["id"]
        group = membership[key]
        if group["split"] not in {"development", "holdout", "prospective"}:
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
    root = Path(root)
    docs, groups, labels, digest = load_corpus(root, reviewed=True)
    locators_path = root / "locators.json"
    locators = json.loads(locators_path.read_text()) if locators_path.exists() else {}
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
        locator = locators.get(key, {})
        source = locator.get("current_url", doc["source"])
        if locator.get("page"):
            source = source.split("#")[0] + f"#page={locator['page']}"
        prior = state.get(event, {})
        fields = prior.get("fields", {}).copy()
        for field, item in labels[key].items():
            fields[field] = {**item, "passage_id": key, "location": doc["location"],
                             "source": source, "notice_date": doc["notice_date"]}
        entry = {"event": event, "symbol": doc["symbol"], "available_at": available.isoformat(),
                 "notice_date": doc["notice_date"], "passage_id": key, "fields": fields,
                 "passages": doc["passages"], "source": source, "locator": locator, "location": doc["location"]}
        state[event] = entry
        changes.append(entry)
    rows = changes if history else list(state.values())
    ref_status = "adjudicated" if (root / "adjudication.json").exists() else "local-catalog"
    return {"clock": clock, "as_of": until.isoformat(), "reference_status": ref_status,
            "availability_basis": ("next Nairobi midnight after the dated issuer disclosure"
                                   if clock == "disclosure" else "conservative bound for this collected snapshot"),
            "corpus_sha256": digest, "rows": [r for r in rows if not status or
                                              r["fields"].get("status", {}).get("value") == status]}


def promote_filing(report, notice_date, catalog_dir):
    """Promote extracted events from a filing-extract report into an append-only local catalog."""
    day = date.fromisoformat(notice_date).isoformat()
    root = Path(catalog_dir)
    root.mkdir(parents=True, exist_ok=True)
    notices = json.loads((root / "notices.json").read_text()) if (root / "notices.json").exists() else []
    membership = json.loads((root / "membership.json").read_text()) if (root / "membership.json").exists() else {}
    reference = json.loads((root / "reference.json").read_text()) if (root / "reference.json").exists() else {}
    locators = json.loads((root / "locators.json").read_text()) if (root / "locators.json").exists() else {}

    symbol = report["symbol"].strip().upper()
    source = report["source"]
    pdf_sha = report.get("pdf_sha256", "")
    by_id = {row["id"]: row for row in notices}
    promoted = []

    for event in report.get("events", []):
        fields = event.get("fields", {})
        passage_id = f"{symbol}-{day}-{event['event_id']}"
        passages, ref_entry = [], {}
        first_page = None
        for name in FIELDS:
            item = fields.get(name)
            if not item or not item.get("citations"):
                continue
            cite = item["citations"][0]
            quote = cite["quote"]
            if quote not in passages:
                passages.append(quote)
            if first_page is None:
                first_page = cite["page"]
            ref_entry[name] = {"value": item["value"], "span": quote}
        if not ref_entry:
            continue
        dist_type = fields.get("distribution_type", {}).get("value")
        period = fields.get("period", {}).get("value")
        event_key = f"{symbol}-{period}-{dist_type}" if (period and dist_type) else None
        location = f"PDF page {first_page}" if first_page else "PDF filing"
        by_id[passage_id] = {
            "id": passage_id, "symbol": symbol, "notice_date": day,
            "source": source, "location": location, "passages": passages,
        }
        membership[passage_id] = {"event": event_key, "split": "prospective"}
        reference[passage_id] = ref_entry
        locators[passage_id] = {"current_url": source, "page": first_page, "pdf_sha256": pdf_sha}
        promoted.append({"id": passage_id, "event": event_key, "fields": sorted(ref_entry)})

    ordered = [by_id[k] for k in sorted(by_id)]
    for name, payload in (("notices", ordered), ("membership", membership),
                          ("reference", reference), ("locators", locators)):
        (root / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    _, _, _, digest = load_corpus(root)
    return {"catalog": str(root), "promoted": promoted, "total_notices": len(ordered), "corpus_sha256": digest}
