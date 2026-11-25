import json
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from asof.cli import get_parser, run as run_cli
from asof.data import load_dataset
from asof.nse_sync import (
    _max_completed_nse_session,
    parse_eodhd_payload,
    parse_stockanalysis_payload,
    sync_nse_universe,
)

NAIROBI = ZoneInfo("Africa/Nairobi")


def test_parse_stockanalysis_and_eodhd_use_raw_close_only():
    sa = parse_stockanalysis_payload({
        "data": [
            {"t": "2026-11-24", "o": 34.0, "h": 34.2, "l": 33.85, "c": 33.95, "a": 31.10, "v": 1435661},
            {"t": "2026-11-23", "o": 33.9, "h": 34.1, "l": 33.80, "c": 34.10, "a": 31.25, "v": 900000},
        ]
    }, "SCOM")
    assert [r["session"] for r in sa] == ["2026-11-23", "2026-11-24"]
    assert sa[1]["close"] == "33.95"  # Raw 'c', NOT pre-adjusted 'a'
    assert sa[1]["price_type"] == "close"

    eod = parse_eodhd_payload([
        {"date": "2026-11-24", "open": 80.0, "high": 81.5, "low": 79.5, "close": 81.0, "adjusted_close": 75.0, "volume": 500000}
    ], "EQTY")
    assert len(eod) == 1
    assert eod[0]["close"] == "81.0"


def test_nairobi_close_guard_blocks_unclosed_intraday_session():
    before_close = datetime(2026, 11, 25, 11, 30, tzinfo=NAIROBI)
    after_close = datetime(2026, 11, 25, 15, 5, tzinfo=NAIROBI)
    assert _max_completed_nse_session(before_close) == "2026-11-24"
    assert _max_completed_nse_session(after_close) == "2026-11-25"


def test_sync_append_only_overlap_reconciliation_and_future_guard(tmp_path):
    uni_dir = tmp_path / "universe"

    # Initial bootstrap sync on 2026-11-24 after close -> admits 2026-11-23 and 2026-11-24, skips 2026-11-25.
    first_payload = {
        "data": [
            {"t": "2026-11-25", "o": 35.0, "h": 35.5, "l": 34.9, "c": 35.2, "a": 35.2, "v": 100000},
            {"t": "2026-11-24", "o": 34.0, "h": 34.2, "l": 33.85, "c": 33.95, "a": 32.0, "v": 1435661},
            {"t": "2026-11-23", "o": 33.9, "h": 34.1, "l": 33.8, "c": 34.10, "a": 32.1, "v": 900000},
        ]
    }
    res1 = sync_nse_universe(
        uni_dir,
        provider="stockanalysis",
        symbols=["SCOM"],
        delay_seconds=0,
        now=datetime(2026, 11, 25, 10, 0, tzinfo=NAIROBI),  # Before 15:00 EAT on Nov 25 -> cutoff Nov 24
        fetcher=lambda _url: json.dumps(first_payload).encode("utf-8"),
    )
    assert res1["cutoff_session"] == "2026-11-24"
    assert res1["bars_added"] == 2
    assert res1["updated"][0]["skipped_future"] == 1
    ds = load_dataset(uni_dir / "SCOM")
    assert [b.session for b in ds.bars] == ["2026-11-23", "2026-11-24"]
    assert ds.meta["sector"] == "Telecommunication"

    # Second sync on Nov 25 after 15:00 EAT with a conflicting Nov 24 close -> must NOT overwrite Nov 24,
    # MUST emit overlap warning, and MUST append Nov 25 cleanly.
    second_payload = {
        "data": [
            {"t": "2026-11-25", "o": 34.0, "h": 34.8, "l": 33.9, "c": 34.50, "a": 34.50, "v": 2000000},
            {"t": "2026-11-24", "o": 34.0, "h": 34.2, "l": 33.85, "c": 36.00, "a": 36.00, "v": 1435661},
        ]
    }
    res2 = sync_nse_universe(
        uni_dir,
        provider="stockanalysis",
        delay_seconds=0,
        now=datetime(2026, 11, 25, 16, 0, tzinfo=NAIROBI),
        fetcher=lambda _url: json.dumps(second_payload).encode("utf-8"),
    )
    assert res2["bars_added"] == 1
    assert res2["overlap_warnings_count"] == 1
    assert res2["overlap_warnings"][0]["session"] == "2026-11-24"
    ds2 = load_dataset(uni_dir / "SCOM")
    assert [b.session for b in ds2.bars] == ["2026-11-23", "2026-11-24", "2026-11-25"]
    assert ds2.bars[1].close == Decimal("33.95")  # Original preserved
    assert ds2.bars[2].close == Decimal("34.50")


def test_eodhd_redacts_api_key_and_csv_url_provider_works(tmp_path):
    uni_eod = tmp_path / "uni_eod"
    seen_urls = []

    def fake_eod(url):
        seen_urls.append(url)
        return json.dumps([
            {"date": "2026-11-24", "open": 80, "high": 82, "low": 79, "close": 81.25, "adjusted_close": 70, "volume": 400000}
        ]).encode("utf-8")

    res = sync_nse_universe(
        uni_eod, provider="eodhd", symbols=["EQTY"], api_key="SUPER_SECRET_TOKEN",
        delay_seconds=0, now=datetime(2026, 11, 25, 10, 0, tzinfo=NAIROBI), fetcher=fake_eod,
    )
    assert res["bars_added"] == 1
    assert "SUPER_SECRET_TOKEN" in seen_urls[0]
    manifest_text = (uni_eod / "EQTY" / "manifest.json").read_text()
    assert "SUPER_SECRET_TOKEN" not in manifest_text  # Secret must never be persisted!

    # Test csv-url provider via CLI-compatible call.
    uni_csv = tmp_path / "uni_csv"
    csv_bytes = (
        "Date,Code,Name,Day Low,Day High,Day Price,Previous,Volume\n"
        "2026-11-24,KCB,KCB Group,84.5,86.0,85.75,85.0,1500000\n"
    ).encode("utf-8")
    csv_res = sync_nse_universe(
        uni_csv, provider="csv-url", csv_url="https://example.test/nse.csv",
        fetcher=lambda _u: csv_bytes,
    )
    assert csv_res["tickers_imported"] == 1
    assert load_dataset(uni_csv / "KCB").bars[0].close == Decimal("85.75")


def test_cli_nse_sync_argument_validation(tmp_path):
    parser = get_parser()
    args = parser.parse_args(["nse-sync", "--universe", str(tmp_path / "u"), "--provider", "eodhd"])
    with pytest.raises(ValueError, match="EODHD_API_KEY"):
        run_cli(args)
