"""Append-only End-of-Day (EOD) NSE Kenya market data sync adapter (pure Python stdlib)."""

import csv
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import load_dataset
from .nse_pricelist import KNOWN_ISSUERS, _normalize_bar_row, _read_csv_map, import_nse_universe

NAIROBI = ZoneInfo("Africa/Nairobi")
PROVIDERS = ("stockanalysis", "eodhd", "csv-url")
STOCKANALYSIS_RANGES = ("1M", "6M", "1Y", "5Y", "MAX")
USER_AGENT = "Mozilla/5.0 (compatible; AsOf-Research/0.6; +https://github.com/ashioyajotham/asof)"
NSE_CLOSE_HOUR = 15


def _dec(value):
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value).strip().replace(",", ""))
        return d if d.is_finite() else None
    except InvalidOperation:
        return None


def _http_get_bytes(url, *, timeout=20, headers=None):
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json,text/csv,*/*"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_stockanalysis_payload(payload, symbol, *, isin=None):
    """Parse StockAnalysis NASE history JSON into normalized raw OHLCV bar dicts (ignoring adjusted 'a')."""
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"unexpected StockAnalysis payload for {symbol}")
    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        raw_day = str(item.get("t") or "").strip()[:10]
        if len(raw_day) != 10:
            continue
        try:
            day = date.fromisoformat(raw_day).isoformat()
        except ValueError:
            continue
        bar = _normalize_bar_row(
            symbol, day,
            isin=isin,
            open_val=_dec(item.get("o")),
            high=_dec(item.get("h")),
            low=_dec(item.get("l")),
            close=_dec(item.get("c")),  # Always raw close 'c', never pre-adjusted 'a'
            prev=None,
            volume=_dec(item.get("v")),
            price_type="close",
            source_page=None,
        )
        if bar is not None:
            out.append(bar)
    out.sort(key=lambda r: r["session"])
    return out


def parse_eodhd_payload(payload, symbol, *, isin=None):
    """Parse EODHD EOD JSON array into normalized raw OHLCV bar dicts (ignoring 'adjusted_close')."""
    if not isinstance(payload, list):
        raise ValueError(f"unexpected EODHD payload for {symbol}")
    out = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_day = str(item.get("date") or "").strip()[:10]
        if len(raw_day) != 10:
            continue
        try:
            day = date.fromisoformat(raw_day).isoformat()
        except ValueError:
            continue
        bar = _normalize_bar_row(
            symbol, day,
            isin=isin,
            open_val=_dec(item.get("open")),
            high=_dec(item.get("high")),
            low=_dec(item.get("low")),
            close=_dec(item.get("close")),  # Always raw close, never adjusted_close
            prev=None,
            volume=_dec(item.get("volume")),
            price_type="close",
            source_page=None,
        )
        if bar is not None:
            out.append(bar)
    out.sort(key=lambda r: r["session"])
    return out


def fetch_stockanalysis_bars(symbol, *, range_code="6M", isin=None, fetcher=_http_get_bytes):
    """Fetch raw NSE Kenya EOD OHLCV bars from StockAnalysis (NASE-{SYMBOL})."""
    if range_code not in STOCKANALYSIS_RANGES:
        raise ValueError(f"range must be one of {', '.join(STOCKANALYSIS_RANGES)}")
    sym = symbol.strip().upper()
    url = f"https://stockanalysis.com/api/symbol/a/NASE-{urllib.parse.quote(sym)}/history?range={range_code}"
    raw = fetcher(url)
    payload = json.loads(raw.decode("utf-8"))
    return parse_stockanalysis_payload(payload, sym, isin=isin), url


def fetch_eodhd_bars(symbol, *, api_key=None, since=None, until=None, isin=None, fetcher=_http_get_bytes):
    """Fetch raw NSE Kenya EOD OHLCV bars from EODHD ({SYMBOL}.NAI)."""
    token = api_key or os.environ.get("EODHD_API_KEY", "").strip()
    if not token:
        raise ValueError("EODHD provider requires --api-key or EODHD_API_KEY in environment")
    sym = symbol.strip().upper()
    params = {"api_token": token, "fmt": "json"}
    if since:
        params["from"] = date.fromisoformat(since).isoformat()
    if until:
        params["to"] = date.fromisoformat(until).isoformat()
    qs = urllib.parse.urlencode(params)
    url = f"https://eodhd.com/api/eod/{urllib.parse.quote(sym)}.NAI?{qs}"
    redacted_url = f"https://eodhd.com/api/eod/{sym}.NAI"
    raw = fetcher(url)
    payload = json.loads(raw.decode("utf-8"))
    return parse_eodhd_payload(payload, sym, isin=isin), redacted_url


def _max_completed_nse_session(now=None):
    """Return the latest date whose NSE daily session (09:00-15:00 EAT) has already closed."""
    current = (now or datetime.now(NAIROBI)).astimezone(NAIROBI)
    if current.hour >= NSE_CLOSE_HOUR:
        return current.date().isoformat()
    return date.fromordinal(current.date().toordinal() - 1).isoformat()


def _resolve_target_symbols(root, symbols):
    if symbols:
        return sorted({s.strip().upper() for s in symbols if s.strip()})
    index_path = root / "manifest.json"
    if index_path.exists():
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        existing = sorted(meta.get("tickers", {}).keys())
        if existing:
            return existing
    return sorted(KNOWN_ISSUERS.keys())


def sync_nse_universe(
    universe_dir,
    *,
    provider="stockanalysis",
    symbols=None,
    range_code="6M",
    since=None,
    until=None,
    api_key=None,
    csv_url=None,
    overwrite=False,
    delay_seconds=0.15,
    now=None,
    fetcher=_http_get_bytes,
):
    """Append completed NSE EOD daily bars from a configured provider into a validated universe directory."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
    if provider == "eodhd" and not (api_key or os.environ.get("EODHD_API_KEY", "").strip()):
        raise ValueError("EODHD provider requires --api-key or EODHD_API_KEY in environment")
    root = Path(universe_dir)
    root.mkdir(parents=True, exist_ok=True)
    captured_dt = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    captured = captured_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    if provider == "csv-url":
        if not csv_url:
            raise ValueError("csv-url provider requires --url")
        raw_csv = fetcher(csv_url)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_csv = Path(tmp) / "nse_sync.csv"
            tmp_csv.write_bytes(raw_csv)
            res = import_nse_universe(
                tmp_csv, root,
                symbols=symbols, source_url=csv_url, ingested_at=captured,
            )
        return {
            "provider": provider,
            "source": csv_url,
            "synced_at": captured,
            **res,
        }

    cutoff_day = _max_completed_nse_session(now=now)
    if until:
        until_day = min(date.fromisoformat(until).isoformat(), cutoff_day)
    else:
        until_day = cutoff_day
    since_day = date.fromisoformat(since).isoformat() if since else None
    if since_day and since_day > until_day:
        raise ValueError("since date must not exceed the latest completed NSE session")

    target_symbols = _resolve_target_symbols(root, symbols)
    index_path = root / "manifest.json"
    existing_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    tickers_meta = dict(existing_index.get("tickers", {}))

    updated_summary = []
    overlap_warnings = []
    errors = []
    total_bars_added = 0

    for idx, sym in enumerate(target_symbols):
        if idx > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        known = KNOWN_ISSUERS.get(sym, {})
        sym_dir = root / sym
        existing_manifest = json.loads((sym_dir / "manifest.json").read_text(encoding="utf-8")) if (sym_dir / "manifest.json").exists() else {}
        existing_isin = known.get("isin")
        if not existing_isin and existing_manifest.get("instrument", "").startswith("XNAI:KE"):
            existing_isin = existing_manifest["instrument"].split(":", 1)[1]

        sessions = _read_csv_map(sym_dir / "sessions.csv", "session")
        prices = _read_csv_map(sym_dir / "prices.csv", "session")
        existing_last = max(sessions) if sessions else None
        effective_since = since_day or existing_last

        try:
            if provider == "stockanalysis":
                bars, src_url = fetch_stockanalysis_bars(
                    sym, range_code=range_code, isin=existing_isin, fetcher=fetcher,
                )
            else:
                bars, src_url = fetch_eodhd_bars(
                    sym, api_key=api_key, since=effective_since, until=until_day,
                    isin=existing_isin, fetcher=fetcher,
                )
        except (urllib.error.URLError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append({"symbol": sym, "error": str(exc)})
            continue

        added = 0
        skipped_future = 0
        for row in bars:
            day = row["session"]
            if day > until_day:
                skipped_future += 1
                continue
            if since_day and day < since_day:
                continue
            if day in prices and not overwrite:
                old_c = _dec(prices[day].get("close"))
                new_c = _dec(row["close"])
                if old_c and new_c and old_c > 0:
                    diff_pct = abs(new_c - old_c) / old_c
                    if diff_pct > Decimal("0.01"):
                        overlap_warnings.append({
                            "symbol": sym, "session": day,
                            "existing_close": format(old_c, "f"),
                            "provider_close": format(new_c, "f"),
                            "diff_pct": round(float(diff_pct) * 100, 2),
                        })
                continue
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
            added += 1

        if added == 0 and not (sym_dir / "manifest.json").exists():
            errors.append({"symbol": sym, "error": "no completed EOD bars returned in window"})
            continue

        if added > 0:
            sym_dir.mkdir(parents=True, exist_ok=True)
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

            instrument = existing_manifest.get("instrument") or (f"XNAI:{existing_isin}" if existing_isin else f"XNAI:{sym}")
            issuer_name = existing_manifest.get("issuer") or known.get("issuer") or sym
            sector_name = existing_manifest.get("sector") or tickers_meta.get(sym, {}).get("sector") or known.get("sector") or "Unclassified"
            manifest = {
                **existing_manifest,
                "adapter": "nse-daily",
                "instrument": instrument,
                "venue": "Nairobi Securities Exchange",
                "issuer": issuer_name,
                "sector": sector_name,
                "currency": "KES",
                "price_basis": "raw",
                "lot_size": existing_manifest.get("lot_size", "1"),
                "calendar_version": f"NSE regular local sessions ({ordered_days[0]} to {ordered_days[-1]})",
                "source": existing_manifest.get("source") or src_url,
                "last_sync_source": src_url,
                "last_synced_at": captured,
                "ingested_at": captured,
                "coverage": existing_manifest.get("coverage") or (
                    f"Raw NSE daily observations synced via {provider} (adjusted close excluded)."
                ),
            }
            (sym_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            load_dataset(sym_dir)
            tickers_meta[sym] = {
                "path": sym,
                "instrument": instrument,
                "issuer": issuer_name,
                "sector": sector_name,
                "sessions": len(ordered_days),
                "first_session": ordered_days[0],
                "last_session": ordered_days[-1],
            }
            total_bars_added += added
            updated_summary.append({
                "symbol": sym,
                "bars_added": added,
                "sessions_total": len(ordered_days),
                "last_session": ordered_days[-1],
                "skipped_future": skipped_future,
            })

    universe_manifest = {
        "venue": "Nairobi Securities Exchange",
        "currency": "KES",
        "source": existing_index.get("source") or f"nse-sync:{provider}",
        "ingested_at": existing_index.get("ingested_at") or captured,
        "last_synced_at": captured,
        "last_sync_provider": provider,
        "tickers": {k: tickers_meta[k] for k in sorted(tickers_meta)},
    }
    index_path.write_text(json.dumps(universe_manifest, indent=2) + "\n")
    return {
        "universe": str(root),
        "provider": provider,
        "cutoff_session": until_day,
        "synced_at": captured,
        "tickers_requested": len(target_symbols),
        "tickers_updated": len(updated_summary),
        "bars_added": total_bars_added,
        "overlap_warnings_count": len(overlap_warnings),
        "overlap_warnings": overlap_warnings[:25],
        "errors_count": len(errors),
        "errors": errors,
        "updated": updated_summary,
    }
