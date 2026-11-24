import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from asof.data import Bar, load_dataset
from asof.paper import Assumptions, PaperEngine
from asof.risk import days_to_exit, sharpe_ratio


def _bar(day, close, volume="10000"):
    stamp = datetime.fromisoformat(f"{day}T15:00:00+03:00").astimezone(timezone.utc)
    c = Decimal(str(close))
    return Bar(day, stamp, stamp, c, c, c, c, Decimal(str(volume)))


def test_stock_split_and_bonus_issue_conserve_nav_with_cash_in_lieu():
    meta = {"instrument": "XNAI:TEST", "lot_size": "1"}
    cfg = Assumptions(cash=Decimal("1000"), fee_bps=Decimal("0"), slippage_bps=Decimal("0"),
                      participation=Decimal("1"), max_exposure=Decimal("1"))
    engine = PaperEngine(meta, "buy_hold", cfg)

    # Day 1: signal buys at Day 2 close (100 KES -> 10 shares, 0 cash).
    engine.observe(_bar("2026-01-05", "100"))
    engine.observe(_bar("2026-01-06", "100"))
    assert engine.units == Decimal("10")
    assert engine.cash == Decimal("0")

    # Day 3: approved 2-for-1 split on ex-date as raw price halves to 50 KES -> 20 shares, NAV unchanged (1000).
    split_doc = {
        "id": "doc-split", "available_at": _bar("2026-01-06", "100").close_time,
        "split": {"id": "SP-1", "kind": "split", "status": "approved",
                  "ex_date": "2026-01-07", "numerator": "2", "denominator": "1"},
    }
    engine.observe(_bar("2026-01-07", "50"), documents=[split_doc])
    assert engine.units == Decimal("20")
    assert engine.ledger[-1]["nav"] == Decimal("1000")
    assert engine.worst == Decimal("0")

    # Day 4: bonus 6-for-5 (20% bonus) on 11 shares? Let's use 1-for-3 bonus (4/3) on 20 shares = 26.666...
    # -> 26 whole shares + 2/3 share cash-in-lieu at 37.50 KES = 25 KES; total NAV = 26*37.50 + 25 = 1000.
    bonus_doc = {
        "id": "doc-bonus", "available_at": _bar("2026-01-07", "50").close_time,
        "split": {"id": "BN-1", "kind": "bonus", "status": "approved",
                  "ex_date": "2026-01-08", "numerator": "4", "denominator": "3"},
    }
    proposed_doc = {
        "id": "doc-prop", "available_at": _bar("2026-01-07", "50").close_time,
        "split": {"id": "SP-PROP", "kind": "split", "status": "proposed",
                  "ex_date": "2026-01-08", "numerator": "10", "denominator": "1"},
    }
    engine.observe(_bar("2026-01-08", "37.50"), documents=[split_doc, bonus_doc, proposed_doc])
    assert engine.units == Decimal("26")
    assert engine.ledger[-1]["nav"] == Decimal("1000")
    report = engine.report()
    assert len(report["splits"]) == 2
    assert report["splits"][1]["cash_in_lieu"] == Decimal("25.00")


def test_nse_loader_validates_curated_split_documents(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "adapter": "nse-daily", "instrument": "XNAI:TEST", "venue": "NSE",
        "issuer": "Test PLC", "currency": "KES", "price_basis": "raw",
        "lot_size": "1", "calendar_version": "test", "source": "test",
        "ingested_at": "2026-01-10T00:00:00Z",
    }))
    (tmp_path / "sessions.csv").write_text(
        "session,open_time,close_time\n2026-01-06,2026-01-06T09:00:00+03:00,2026-01-06T15:00:00+03:00\n"
    )
    (tmp_path / "prices.csv").write_text(
        "session,open,high,low,close,volume,status,price_type,available_at\n"
        "2026-01-06,10,10,10,10,1000,trading,close,2026-01-06T15:00:00+03:00\n"
    )
    (tmp_path / "documents.json").write_text(json.dumps([{
        "id": "s1", "source": "https://example.test", "issuer": "Test PLC",
        "kind": "bonus_notice", "attribution": "issuer", "pages": ["Bonus 1 for 5"],
        "published": "2026-01-05T10:00:00Z", "ingested_at": "2026-01-05T11:00:00Z",
        "split": {"id": "B1", "kind": "bonus", "status": "approved",
                  "ex_date": "2026-01-06", "numerator": "6", "denominator": "5"},
    }]))
    ds = load_dataset(tmp_path)
    assert ds.documents[0]["split"]["numerator"] == "6"

    # Invalid numerator <= 0 must raise.
    bad = json.loads((tmp_path / "documents.json").read_text())
    bad[0]["split"]["numerator"] = "0"
    (tmp_path / "documents.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="invalid curated split"):
        load_dataset(tmp_path)


def test_sharpe_ratio_and_days_to_exit_metrics():
    assert sharpe_ratio([100, 100, 100]) is None
    assert sharpe_ratio([100, 101]) is None
    s = sharpe_ratio([100, 102, 101, 104, 107])
    assert s is not None and s > 0

    assert days_to_exit(0, [1000, 2000], 0.1) == 0.0
    assert days_to_exit(500, [0, 0], 0.1) is None
    assert days_to_exit(500, [1000, 1000], 0.1) == pytest.approx(5.0)
