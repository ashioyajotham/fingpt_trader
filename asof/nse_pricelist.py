"""Parse NSE Daily Equity Price Lists (PDF or CSV) into point-in-time nse-daily datasets."""
import csv
import json
import re
import shlex
import subprocess
import tempfile
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


def _parse_date_string(text):
    for pattern, fmt in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return datetime.strptime(match[1], fmt).date().isoformat()
            except ValueError:
                continue
    return None


def extract_session_date(pages, *, filename=None):
    """Find the trading session date from the price list header or filename."""
    candidates = [line for text in pages[:2] for line in text.splitlines()[:20]]
    if filename:
        candidates.append(str(filename))
    for line in candidates:
        parsed = _parse_date_string(line)
        if parsed:
            return parsed
    raise ValueError("could not detect trading session date from NSE price list header; pass --session YYYY-MM-DD")


def _parse_decimal(token):
    if token is None:
        return None
    cleaned = str(token).strip().replace(",", "")
    if not cleaned or cleaned in {"-", "--", "N/A", "n/a"}:
        return None
    try:
        return Decimal(cleaned)
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


def _normalize_bar_row(symbol, session, *, isin, open_val, high, low, close, prev, volume, price_type, source_page):
    vol = volume if volume is not None and volume >= 0 else Decimal(0)
    mark = close if close and close > 0 else prev
    if mark is None or mark <= 0:
        return None
    op = open_val if open_val and open_val > 0 else None
    marks = [mark] if op is None else [op, mark]
    hi = high if high and high >= max(marks) else max(marks)
    lo = low if low and Decimal(0) < low <= min(marks) else min(marks)
    resolved_type = "close" if (op is not None and price_type != "vwap") else "vwap"
    return {
        "symbol": symbol,
        "isin": isin,
        "session": session,
        "open": "" if resolved_type == "vwap" else format(op, "f"),
        "high": format(hi, "f"),
        "low": format(lo, "f"),
        "close": format(mark, "f"),
        "volume": format(vol, "f"),
        "status": "trading",
        "price_type": resolved_type,
        "source_page": source_page,
    }


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
            if len(numeric) < 5:
                continue
            high, low, vwap, prev, volume = (_parse_decimal(t) for t in numeric[:5])
            row_isin = isin or (m[1] if (m := ISIN_RE.search(line)) else KNOWN_ISSUERS.get(symbol, {}).get("isin"))
            row = _normalize_bar_row(
                symbol, session, isin=row_isin, open_val=None, high=high, low=low,
                close=vwap, prev=prev, volume=volume, price_type="vwap", source_page=page_num,
            )
            if row is not None:
                return row
    raise ValueError(f"could not locate an NSE price row for {symbol} on {session}")


def _col(row_map, *names):
    for name in names:
        if name in row_map:
            return row_map[name]
    return None


def _norm_header(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _extract_csv_row(norm, *, symbol, default_isin, fallback_day, csv_path):
    raw_date = _col(norm, "session", "date", "trade_date", "trading_date")
    day = _parse_date_string(raw_date) if raw_date else fallback_day
    if not day:
        raise ValueError(f"missing session date in CSV {csv_path}; pass --session YYYY-MM-DD")
    row_isin = _col(norm, "isin") or default_isin
    op = _parse_decimal(_col(norm, "open", "open_price"))
    hi = _parse_decimal(_col(norm, "high", "day_high", "day_highprice", "day_s_high_price"))
    lo = _parse_decimal(_col(norm, "low", "day_low", "day_lowprice", "day_s_low_price"))
    vwap_raw = _col(norm, "vwap")
    close_raw = _col(norm, "close", "closing_price", "day_price", "day_final_price", "day_s_final_price", "last", "price")
    close_val = _parse_decimal(close_raw if close_raw is not None else vwap_raw)
    prev_val = _parse_decimal(_col(norm, "prev", "previous", "prev_close", "previous_price", "previous_traded_price"))
    vol = _parse_decimal(_col(norm, "volume", "volume_traded", "shares", "shares_traded", "qty"))
    ptype = "vwap" if (vwap_raw is not None and close_raw is None) or op is None else "close"
    return _normalize_bar_row(
        symbol, day, isin=row_isin, open_val=op, high=hi, low=lo,
        close=close_val, prev=prev_val, volume=vol, price_type=ptype, source_page=None,
    )


def parse_nse_csv(csv_path, symbol, *, isin=None, session_date=None):
    """Parse a single-ticker or multi-ticker NSE daily price list CSV into bar observations."""
    symbol = symbol.strip().upper()
    default_isin = isin or KNOWN_ISSUERS.get(symbol, {}).get("isin")
    fallback_day = date.fromisoformat(session_date).isoformat() if session_date else _parse_date_string(Path(csv_path).name)
    rows = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV file: {csv_path}")
        headers = {_norm_header(h): h for h in reader.fieldnames if h}
        has_id_col = any(k in headers for k in ("symbol", "ticker", "code", "stock_code", "security", "issuer", "name", "stock_name", "isin"))
        for raw in reader:
            norm = {k: (raw[orig] or "").strip() for k, orig in headers.items()}
            if has_id_col:
                identity = " ".join(filter(None, (
                    _col(norm, "symbol", "ticker", "code", "stock_code"),
                    _col(norm, "security", "issuer", "name", "stock_name"),
                    _col(norm, "isin"),
                )))
                if not _row_matches(identity, symbol, isin):
                    continue
            bar = _extract_csv_row(norm, symbol=symbol, default_isin=default_isin, fallback_day=fallback_day, csv_path=csv_path)
            if bar is not None:
                rows.append(bar)
    if not rows:
        raise ValueError(f"no matching NSE price rows for {symbol} in {csv_path}")
    return rows


def import_nse_universe(input_path, out_dir, *, symbols=None, source_url=None, ingested_at=None):
    """Bulk-import multi-ticker NSE All-Stocks CSV(s) into a validated multi-ticker universe directory."""
    src = Path(input_path)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    if not files:
        raise ValueError(f"no CSV files found at {src}")
    wanted = {s.strip().upper() for s in symbols} if symbols else None
    captured = ingested_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    grouped = {}
    issuers = {}
    for file in files:
        fallback_day = _parse_date_string(file.name)
        with file.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                continue
            headers = {_norm_header(h): h for h in reader.fieldnames if h}
            for raw in reader:
                norm = {k: (raw[orig] or "").strip() for k, orig in headers.items()}
                sym = (_col(norm, "code", "stock_code", "symbol", "ticker") or "").upper()
                if not sym or not re.fullmatch(r"[A-Z0-9.-]+", sym):
                    continue
                if wanted and sym not in wanted:
                    continue
                name = _col(norm, "name", "stock_name", "security", "issuer")
                if name and sym not in issuers:
                    issuers[sym] = name
                known = KNOWN_ISSUERS.get(sym, {})
                bar = _extract_csv_row(
                    norm, symbol=sym, default_isin=known.get("isin"),
                    fallback_day=fallback_day, csv_path=file,
                )
                if bar is not None:
                    grouped.setdefault(sym, {})[bar["session"]] = bar

    if not grouped:
        raise ValueError(f"no valid NSE equity rows found in {src}")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "manifest.json"
    existing_index = json.loads(index_path.read_text()) if index_path.exists() else {}
    tickers_meta = existing_index.get("tickers", {})

    for sym, new_bars in sorted(grouped.items()):
        known = KNOWN_ISSUERS.get(sym, {})
        sym_dir = root / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        sessions = _read_csv_map(sym_dir / "sessions.csv", "session")
        prices = _read_csv_map(sym_dir / "prices.csv", "session")
        detected_isin = known.get("isin")
        for day, row in sorted(new_bars.items()):
            detected_isin = detected_isin or row["isin"]
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
        ordered_days = sorted(sessions)
        with (sym_dir / "sessions.csv").open("w", encoding="utf-8", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=["session", "open_time", "close_time"])
            w.writeheader()
            for day in ordered_days:
                w.writerow(sessions[day])
        with (sym_dir / "prices.csv").open("w", encoding="utf-8", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=[
                "session", "open", "high", "low", "close", "volume", "status", "price_type", "available_at",
            ])
            w.writeheader()
            for day in ordered_days:
                w.writerow(prices[day])
        docs_path = sym_dir / "documents.json"
        if not docs_path.exists():
            docs_path.write_text("[]\n")
        instrument = f"XNAI:{detected_isin}" if detected_isin else f"XNAI:{sym}"
        issuer_name = known.get("issuer") or issuers.get(sym) or sym
        manifest = {
            "adapter": "nse-daily",
            "instrument": instrument,
            "venue": "Nairobi Securities Exchange",
            "issuer": issuer_name,
            "currency": "KES",
            "price_basis": "raw",
            "lot_size": "1",
            "calendar_version": f"NSE regular local sessions ({ordered_days[0]} to {ordered_days[-1]})",
            "source": source_url or str(src.resolve()),
            "ingested_at": captured,
            "coverage": "Raw NSE daily observations imported from multi-ticker CSV (Adjusted Price excluded).",
        }
        (sym_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_dataset(sym_dir)
        tickers_meta[sym] = {
            "path": sym,
            "instrument": instrument,
            "issuer": issuer_name,
            "sessions": len(ordered_days),
            "first_session": ordered_days[0],
            "last_session": ordered_days[-1],
        }

    universe_manifest = {
        "venue": "Nairobi Securities Exchange",
        "currency": "KES",
        "source": source_url or str(src.resolve()),
        "ingested_at": captured,
        "tickers": {k: tickers_meta[k] for k in sorted(tickers_meta)},
    }
    index_path.write_text(json.dumps(universe_manifest, indent=2) + "\n")
    return {
        "universe": str(root),
        "tickers_imported": len(grouped),
        "tickers_total": len(universe_manifest["tickers"]),
        "symbols": sorted(grouped),
    }


def _read_csv_map(path, key_field):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row[key_field]: row for row in csv.DictReader(stream)}


def _read_pdf_with_optional_ocr(file, ocr_cmd=None):
    pages, meta = read_pdf(file)
    if not all(p["status"] == "unreadable" for p in meta["coverage"]["pages"]):
        return pages, meta
    if not ocr_cmd:
        raise ValueError(
            f"{file.name} has no selectable text layer (scanned/image-only PDF); "
            "pass --ocr-cmd, run OCR first, or supply a CSV / selectable-text PDF"
        )
    with tempfile.TemporaryDirectory() as tmp:
        ocr_out = Path(tmp) / f"ocr-{file.name}"
        cmd = [part.replace("{in}", str(file)).replace("{out}", str(ocr_out)) for part in shlex.split(ocr_cmd)]
        subprocess.run(cmd, check=True)
        pages, meta = read_pdf(ocr_out)
        if all(p["status"] == "unreadable" for p in meta["coverage"]["pages"]):
            raise ValueError(f"OCR command produced no selectable text for {file.name}")
        return pages, meta


def import_nse_pricelist(input_path, symbol, out_dir, *, issuer=None, isin=None,
                         session_date=None, source_url=None, ingested_at=None, ocr_cmd=None):
    """Parse PDF or CSV NSE Daily Price List file(s) and append to an nse-daily dataset."""
    src = Path(input_path)
    if src.is_dir():
        files = sorted([*src.glob("*.pdf"), *src.glob("*.csv")])
    else:
        files = [src]
    if not files:
        raise ValueError(f"no PDF or CSV files found at {src}")
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
        if file.suffix.lower() == ".csv":
            parsed_rows = parse_nse_csv(
                file, symbol, isin=detected_isin,
                session_date=session_date if len(files) == 1 else None,
            )
        else:
            pages, meta = _read_pdf_with_optional_ocr(file, ocr_cmd=ocr_cmd)
            parsed_rows = [parse_nse_pricelist_text(
                pages, symbol, isin=detected_isin,
                session_date=session_date if len(files) == 1 else None,
                filename=file.name,
            )]
            last_sha = meta["pdf_sha256"]
        for row in parsed_rows:
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
        "coverage": "Imported from NSE daily price list file(s). Opening price is preserved when present, else VWAP mark.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_dataset(out)
    return {"dataset": str(out), "instrument": instrument, "sessions_total": len(ordered_days), "imported": imported}
