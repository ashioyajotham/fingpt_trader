"""Import one privately supplied NSE listing and an explicit calendar of sessions."""

import csv
import json
from datetime import date
from zoneinfo import ZoneInfo

from .data import Bar, instant, number


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_nse(root, meta):
    if meta["currency"] != "KES":
        raise ValueError("NSE daily importer expects KES prices")
    calendar = {}
    nairobi = ZoneInfo("Africa/Nairobi")
    for row in read_csv(root / "sessions.csv"):
        day = date.fromisoformat(row["session"]).isoformat()
        opening, closing = instant(row["open_time"]), instant(row["close_time"])
        if (opening >= closing or day in calendar or
                any(t.astimezone(nairobi).date().isoformat() != day for t in (opening, closing))):
            raise ValueError(f"invalid or duplicate NSE session: {day}")
        calendar[day] = opening, closing
    bars = []
    for row in read_csv(root / "prices.csv"):
        day = row["session"]
        close_time = calendar[day][1]
        bars.append(Bar(day, close_time, instant(row["available_at"]),
                        *(number(row[k]) for k in ("open", "high", "low", "close", "volume")),
                        status=row["status"]))
    if [b.session for b in bars] != sorted(calendar):
        raise ValueError("each calendar session requires a raw price row, including suspensions")
    documents = json.loads((root / "documents.json").read_text(encoding="utf-8"))
    ids = set()
    for doc in documents:
        if doc["id"] in ids or not all(doc[k] for k in ("id", "source", "issuer", "kind", "pages", "attribution")):
            raise ValueError("duplicate or incomplete source document")
        if doc["issuer"] != meta["issuer"] or not all(isinstance(p, str) and p.strip() for p in doc["pages"]):
            raise ValueError("issuer or source pages invalid")
        ids.add(doc["id"])
        published = doc["published"]
        if len(published) == 10:
            day = date.fromisoformat(published).isoformat()
            following = [opening for session, (opening, _) in calendar.items() if session > day]
            if not following:
                raise ValueError(f"extend session calendar beyond {day}")
            doc["available_at"] = min(following)
            doc["availability_basis"] = "date only; next listed session open"
        else:
            doc["available_at"] = instant(published)
            doc["availability_basis"] = "timestamped publication"
        received = instant(doc["ingested_at"])
        if received > instant(meta["ingested_at"]):
            raise ValueError("document ingestion after dataset capture")
        if len(published) != 10 and received < doc["available_at"]:
            raise ValueError("document ingested before publication")
        if "dividend" in doc:
            action = doc["dividend"]
            if doc["kind"] != "dividend_notice" or action["currency"] != meta["currency"]:
                raise ValueError("dividend source or currency mismatch")
            if (not action["id"] or number(action["amount"]) < 0 or
                    action["status"] not in {"proposed", "approved", "cancelled"} or
                    date.fromisoformat(action["pay_date"]) < date.fromisoformat(action["ex_date"])):
                raise ValueError("invalid curated dividend terms")
    documents.sort(key=lambda d: (d["available_at"], d["id"]))
    seen = {}
    predecessors = {}
    for doc in documents:
        if "supersedes" in doc:
            previous = predecessors.get(doc["supersedes"])
            if previous is None or previous["available_at"] >= doc["available_at"]:
                raise ValueError("document revision requires an earlier source")
        predecessors[doc["id"]] = doc
        if "dividend" in doc:
            key = doc["dividend"]["id"]
            if seen.get(key) == doc["available_at"]:
                raise ValueError(f"ambiguous simultaneous dividend revisions: {key}")
            seen[key] = doc["available_at"]
    return bars, documents
