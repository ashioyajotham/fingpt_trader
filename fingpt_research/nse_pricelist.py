"""Parse standard NSE Daily Equity Price List PDFs into point-in-time nse-daily datasets."""
import csv
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .data import load_dataset
from .filing_pdf import read_pdf

KNOWN_ISSUERS = {
    "SCOM": {"isin": "KE1000001402", "issuer": "Safaricom PLC", "aliases": ("safaricom", "scom")},
    "EQTY": {"isin": "KE0000000554", "issuer": "Equity Group Holdings PLC", "aliases": ("equity group", "eqty")},
    "KCB": {"isin": "KE0000000315", "issuer": "KCB Group PLC", "aliases": ("kcb group", "kcb")},
    "SCBK": {"isin": "KE0000000448", "issuer": "Standard Chartered Bank Kenya Ltd", "aliases": ("standard chartered", "scbk")},
    "EABL": {"isin": "KE0000000216", "issuer": "East African Breweries PLC", "aliases": ("east african breweries", "eabl")},
    "BAT": {"isin": "KE0000000075", "issuer": "British American Tobacco Kenya PLC", "aliases": ("british american tobacco", "bat kenya", "bat")},
    "KPLC": {"isin": "KE0000000349", "issuer": "Kenya Power & Lighting Co PLC", "aliases": ("kenya power", "kplc")},
    "KEGN": {"isin": "KE0000000547", "issuer": "Kenya Electricity Generating Co PLC", "aliases": ("kengen", "kenya electricity generating", "kegn")},
    "IMH": {"isin": "KE0000000604", "issuer": "I&M Group PLC", "aliases": ("i&m group", "i&m holdings", "imh")},
    "ABSA": {"isin": "KE0000000067", "issuer": "Absa Bank Kenya PLC", "aliases": ("absa bank kenya", "absa")},
    "COOP": {"isin": "KE1000001568", "issuer": "Co-operative Bank of Kenya Ltd", "aliases": ("co-operative bank", "coop")},
    "NCBA": {"isin": "KE0000000406", "issuer": "NCBA Group PLC", "aliases": ("ncba group", "ncba")},
}

DATE_PATTERNS = (
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{1,2}-[A-Za-z]{3}-\d{4})\b"), "%d-%b-%Y"),
    (re.compile(r"\b(\d{1,2}-[A-Za-z]{3}-\d{2})\b"), "%d-%b-%y"),
    (re.compile(r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b"), "%d %B %Y"),
    (re.compile(r"\b(\d{1,2}/\d{2}/\d{4})\b"), "%d/%m/%Y"),
)
ISIN_RE = re.compile(r"\b(KE\d{10}|[A-Z]{2}[A-Z0-9]{10})\b")
NUM_TOKEN = re.compile(r"^(?:[\d,]+(?:\.\d+)?|-)$")


def extract_session_date(pages, *, filename=None):
    """Find the trading session date from the price list header or filename."""
    candidates = [line for text in pages[:2] for line in text.splitlines()[:20]]
    if filename:
        candidates.append(str(filename))
    for line in candidates:
        for pattern, fmt in DATE_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    return datetime.strptime(match[1], fmt).date().isoformat()
                except ValueError:
                    continue
    raise ValueError("could not detect trading session date from NSE price list header; pass --session YYYY-MM-DD")


def _parse_decimal(token):
    if token == "-":
        return None
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def _row_matches(line, symbol, isin=None):
    low = line.casefold()
    info = KNOWN_ISSUERS.get(symbol, {})
    target_isin = isin or info.get("isin")
    if target_isin and target_isin.casefold() in low:
        return True
    if re.search(rf"\b{re.escape(symbol.casefold())}\b", low):
        return True
    return any(alias in low for alias in info.get("aliases", ()))


def parse_nse_pricelist_text(pages, symbol, *, isin=None, session_date=None, filename=None):
    """Extract one equity's daily High/Low/VWAP/Prev/Volume observation from NSE Daily Price List text."""
    symbol = symbol.strip().upper()
    session = date.fromisoformat(session_date).isoformat() if session_date else extract_session_date(pages, filename=filename)
    for page_num, text in enumerate(pages, 1):
        for raw in text.splitlines():
            line = raw.strip()
            if not line or not _row_matches(line, symbol, isin):
                continue
            tokens = line.split()
            numeric = [t for t in tokens if NUM_TOKEN.match(t)]
            # Standard NSE Daily Equity Price List numeric columns at row end:
            # High, Low, VWAP (Close), Previous, Volume (optionally followed by additional cols).
            if len(numeric) < 5:
                continue
            # Take the first 5 numeric tokens after any par-value/code prefix (preferring the trailing 5-tuple
            # when exactly 5, or first 5 price/vol fields: High, Low, VWAP, Prev, Volume).
            high, low, vwap, prev, volume = (_parse_decimal(t) for t in numeric[:5])
            row_isin = isin or (m[1] if (m := ISIN_RE.search(line)) else KNOWN_ISSUERS.get(symbol, {}).get("isin"))
            vol = volume if volume is not None and volume >= 0 else Decimal(0)
            mark = vwap if vwap and vwap > 0 else prev
            if mark is None or mark <= 0:
                continue
            hi = high if high and high >= mark else mark
            lo = low if low and Decimal(0) < low <= mark else mark
            if lo > hi:
                lo, hi = min(lo, mark), max(hi, mark)
            return {
                "symbol": symbol,
                "isin": row_isin,
                "session": session,
                "open": "",
                "high": format(hi, "f"),
                "low": format(lo, "f"),
                "close": format(mark, "f"),
                "volume": format(vol, "f"),
                "status": "trading",
                "price_type": "vwap",
                "source_page": page_num,
            }
    raise ValueError(f"could not locate an NSE price row for {symbol} on {session}")


def _read_csv_map(path, key_field):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row[key_field]: row for row in csv.DictReader(stream)}


def import_nse_pricelist(pdf_path, symbol, out_dir, *, issuer=None, isin=None,
                         session_date=None, source_url=None, ingested_at=None):
    """Parse one PDF or a directory of NSE Daily Price List PDFs and append to an nse-daily dataset."""
    src = Path(pdf_path)
    files = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    if not files:
        raise ValueError(f"no PDF files found at {src}")
    symbol = symbol.strip().upper()
    known = KNOWN_ISSUERS.get(symbol, {})
    captured = ingested_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = _read_csv_map(out / "sessions.csv", "session")
    prices = _read_csv_map(out / "prices.csv", "session")

    detected_isin = isin or known.get("isin")
    imported = []
    last_sha = ""
    last_page = None
    for file in files:
        pages, meta = read_pdf(file)
        if all(p["status"] == "unreadable" for p in meta["coverage"]["pages"]):
            raise ValueError(
                f"{file.name} has no selectable text layer (scanned/image-only PDF); "
                "run OCR first or supply a selectable-text NSE price list PDF"
            )
        row = parse_nse_pricelist_text(
            pages, symbol, isin=detected_isin,
            session_date=session_date if len(files) == 1 else None,
            filename=file.name,
        )
        detected_isin = detected_isin or row["isin"]
        day = row["session"]
        sessions[day] = {
            "session": day,
            "open_time": f"{day}T09:00:00+03:00",
            "close_time": f"{day}T15:00:00+03:00",
        }
        prices[day] = {
            "session": day,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "status": row["status"],
            "price_type": row["price_type"],
            "available_at": f"{day}T15:00:00+03:00",
        }
        last_sha = meta["pdf_sha256"]
        last_page = row["source_page"]
        imported.append({"file": file.name, "session": day, "close": row["close"], "volume": row["volume"]})

    ordered_days = sorted(sessions)
    with (out / "sessions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session", "open_time", "close_time"])
        writer.writeheader()
        for day in ordered_days:
            writer.writerow(sessions[day])
    with (out / "prices.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "session", "open", "high", "low", "close", "volume", "status", "price_type", "available_at",
        ])
        writer.writeheader()
        for day in ordered_days:
            writer.writerow(prices[day])

    docs_path = out / "documents.json"
    if not docs_path.exists():
        docs_path.write_text("[]\n")

    instrument = f"XNAI:{detected_isin}" if detected_isin else f"XNAI:{symbol}"
    manifest = {
        "adapter": "nse-daily",
        "instrument": instrument,
        "venue": "Nairobi Securities Exchange",
        "issuer": issuer or known.get("issuer") or symbol,
        "currency": "KES",
        "price_basis": "raw",
        "lot_size": "1",
        "calendar_version": f"NSE regular local sessions ({ordered_days[0]} to {ordered_days[-1]})",
        "source": source_url or str(src.resolve()),
        "source_sha256": last_sha,
        "source_page": last_page,
        "ingested_at": captured,
        "coverage": "Imported from NSE daily price list PDF(s). VWAP is a valuation mark; opening price is unavailable.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Validate the resulting dataset through the strict point-in-time loader.
    load_dataset(out)
    return {"dataset": str(out), "instrument": instrument, "sessions_total": len(ordered_days), "imported": imported}
