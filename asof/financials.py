"""Point-in-time persisted financial-statement catalog and --as-of queries."""
import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from .dividends import NAIROBI, cutoff
from .data import instant

FINANCIAL_METRICS = (
    "revenue", "profit_before_tax", "profit_after_tax", "eps",
    "total_assets", "total_equity", "loans_and_advances", "customer_deposits",
)


def _catalog_digest(path):
    raw = path.read_bytes()
    return hashlib.sha256(b"financials\0" + raw).hexdigest()


def load_financials(root):
    """Load and validate a local point-in-time financials catalog."""
    path = Path(root) / "financials.json"
    if not path.exists():
        return [], hashlib.sha256(b"financials\0[]").hexdigest()
    rows = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for row in rows:
        key = row["id"]
        if key in seen:
            raise ValueError(f"duplicate financials record: {key}")
        seen.add(key)
        date.fromisoformat(row["notice_date"])
        if not row.get("symbol") or not row.get("source"):
            raise ValueError(f"incomplete financials provenance for {key}")
        for name, item in row.get("metrics", {}).items():
            if name not in FINANCIAL_METRICS:
                raise ValueError(f"unknown financial metric {name} in {key}")
            cites = item.get("citations") or []
            if not cites or any(not c.get("quote") or int(c.get("page", 0)) < 1 for c in cites):
                raise ValueError(f"missing verbatim citation for {key} {name}")
    return rows, _catalog_digest(path)


def promote_financials(report, notice_date, catalog_dir, *, ingested_at=None):
    """Promote cited financial-statement figures from a filing report into catalog_dir/financials.json."""
    day = date.fromisoformat(notice_date).isoformat()
    root = Path(catalog_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "financials.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {r["id"]: r for r in rows}

    symbol = report["symbol"].strip().upper()
    source = report["source"]
    pdf_sha = report.get("pdf_sha256", "")
    captured = ingested_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    raw_metrics = report.get("financials", {})
    if not raw_metrics:
        raise ValueError("report contains no extracted financial metrics to promote")

    by_period = {}
    for name, item in raw_metrics.items():
        if name not in FINANCIAL_METRICS:
            continue
        period = str(item.get("period") or "unknown")
        by_period.setdefault(period, {})[name] = item

    promoted = []
    for period, metric_map in sorted(by_period.items()):
        rec_id = f"{symbol}-{day}-{period}"
        by_id[rec_id] = {
            "id": rec_id,
            "symbol": symbol,
            "period": period,
            "notice_date": day,
            "ingested_at": captured,
            "source": source,
            "pdf_sha256": pdf_sha,
            "metrics": metric_map,
        }
        promoted.append({"id": rec_id, "period": period, "metrics": sorted(metric_map)})

    ordered = [by_id[k] for k in sorted(by_id)]
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _, digest = load_financials(root)
    return {"catalog": str(root), "promoted": promoted, "total_records": len(ordered), "catalog_sha256": digest}


def query_financials(as_of, *, root, symbol=None, metric=None, history=False, clock="disclosure"):
    """Query financial statements strictly visible as of the given Nairobi date or ISO timestamp."""
    rows, digest = load_financials(root)
    until = cutoff(as_of)
    state, changes = {}, []
    for rec in sorted(rows, key=lambda r: (r["notice_date"], r["id"])):
        next_day = date.fromisoformat(rec["notice_date"]) + timedelta(days=1)
        available = datetime.combine(next_day, time.min, NAIROBI)
        if clock == "observed":
            available = max(instant(rec["ingested_at"]), available)
        if available > until:
            continue
        if symbol and rec["symbol"] != symbol.upper():
            continue
        group_key = f"{rec['symbol']}-{rec['period']}"
        prior = state.get(group_key, {})
        merged = prior.get("metrics", {}).copy()
        for name, item in rec["metrics"].items():
            if metric and name != metric:
                continue
            merged[name] = {
                **item,
                "record_id": rec["id"],
                "notice_date": rec["notice_date"],
                "source": rec["source"],
            }
        if not merged:
            continue
        entry = {
            "key": group_key,
            "symbol": rec["symbol"],
            "period": rec["period"],
            "available_at": available.isoformat(),
            "notice_date": rec["notice_date"],
            "record_id": rec["id"],
            "source": rec["source"],
            "pdf_sha256": rec.get("pdf_sha256", ""),
            "metrics": merged,
        }
        state[group_key] = entry
        changes.append(entry)
    result_rows = changes if history else list(state.values())
    return {
        "clock": clock,
        "as_of": until.isoformat(),
        "availability_basis": ("next Nairobi midnight after the dated issuer disclosure"
                               if clock == "disclosure" else "conservative bound for this collected snapshot"),
        "catalog_sha256": digest,
        "rows": result_rows,
    }
