import json
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

import pytest

from asof.borrow import BorrowSchedule
from asof.data import FIXTURES, instant, load_dataset
from asof.experiments import simulate
from asof.forward import ForwardBook
from asof.paper import Assumptions, PaperEngine
from asof.portfolio import FX, portfolio

SPEC = Path(__file__).resolve().parents[1] / "examples/kes-portfolio.json"


def test_public_mark_has_no_invented_open_or_fill():
    ds = load_dataset(FIXTURES / "nse-public-sample")
    sample = ds.bars[0]
    assert sample.open is None and sample.price_type == "vwap" and sample.close == D("38.65")
    assert ds.report(instant("2026-09-02T15:00:00+03:00"))["observations"] == []
    earlier = replace(sample, session="2026-09-01", available_at=instant("2026-09-01T12:00:00Z"),
                      close_time=instant("2026-09-01T12:00:00Z"), price_type="close", open=sample.close)
    engine = PaperEngine(ds.meta, "buy_hold", Assumptions())
    engine.observe(earlier)
    assert engine.pending.quantity > 0
    engine.observe(sample)
    assert engine.fills == []


def test_cash_sleeves_translate_at_known_fx_and_fail_on_stale(tmp_path):
    spec = json.loads(SPEC.read_text())
    for row in spec["sleeves"]:
        row["policy"] = "cash"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(spec))
    result = portfolio(path)
    assert result["ledger"][0]["nav"] == 100000
    assert result["ledger"][-1]["nav"] == 50000 + D(50000) / 130 * 135
    spec["max_fx_age_hours"] = 1
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="stale FX"):
        portfolio(path)


def test_fx_unobservable_until_available():
    fx = FX("KES", [dict(currency="USD", observed_at="2025-01-01T10:00:00Z",
                         available_at="2025-01-02T09:00:00Z", base_per_unit=130, source="fixture")], 48)
    with pytest.raises(ValueError, match="missing FX"):
        fx.rate("USD", instant("2025-01-02T08:59:59Z"))
    assert fx.rate("USD", instant("2025-01-02T09:00:00Z")) == 130


def test_borrow_fees_recall_dividend_and_forward_parity():
    spec = json.loads(SPEC.read_text())
    ds = load_dataset(FIXTURES / "nse")
    borrow = BorrowSchedule(ds.meta["instrument"], spec["sleeves"][1]["borrow"])
    cfg = Assumptions(fee_bps=D(0), slippage_bps=D(0), participation=D(1))
    history = simulate(ds, "short_hold", cfg, borrow=borrow)
    first = ForwardBook(replace(ds, bars=ds.bars[:2]), "short_hold", cfg, [], borrow)
    first.advance(ds, [])
    assert first.engine.report() == history.report()
    assert len(history.fills) == 2
    sale, cover = history.fills
    assert (sale["quantity"], sale["session"]) == (-100, "2025-01-03")
    assert (cover["quantity"], cover["reason"], cover["session"]) == (100, "forced_buy_in", "2025-01-13")
    assert cover["price"] == D(101) * D("1.0025")
    # Three calendar days from Friday to Monday at the Friday closing mark.
    assert history.ledger[2]["borrow_fee"] == D(9900) * D(".12") * 3 / 365
    assert sum(v["amount"] for v in history.cash_events) == -120


def test_locate_timing_expiry_and_partial_buy_in():
    ds = load_dataset(FIXTURES / "nse")
    row = json.loads(SPEC.read_text())["sleeves"][1]["borrow"][0]
    row.update(available_at=ds.bars[1].close_time.isoformat(), expires_at=ds.bars[3].close_time.isoformat())
    loan = BorrowSchedule(ds.meta["instrument"], [row])
    cfg = Assumptions(participation=D("1"))
    book = PaperEngine(ds.meta, "short_hold", cfg, loan)
    for bar in ds.bars[:3]:
        book.observe(bar)
    assert len(book.fills) == 1 and book.fills[0]["session"] == ds.bars[2].session
    book.observe(replace(ds.bars[3], volume=D(35)))
    assert book.units == -65 and book.fills[-1]["reason"] == "forced_buy_in"
    book.observe(ds.bars[4])
    assert book.units == 0
    with pytest.raises(ValueError, match="requires"):
        PaperEngine(ds.meta, "short_hold", cfg)


def test_evaluation_accepts_another_backend_with_shared_citation_checks():
    from asof.evidence import EvidenceModel, evaluate_documents

    class RecordedCandidate(EvidenceModel):
        name = "recorded"
        provider = "test"
        version = "1"
        diagnostics = {}
        tokens = 0

        def passages(self, doc):
            yield from enumerate(doc["pages"], 1)

        def generate_json(self, instruction, doc, schema=None):
            assert schema["properties"]["amount"]
            return {"amount": {"value": "3", "page": 1, "quote": "3 per share"},
                    "issuer": {"value": "made up", "page": 1, "quote": "made up"}}

    root = FIXTURES / "issuer-excerpts"
    result = evaluate_documents(json.loads((root / "documents.json").read_text())[:1],
                                json.loads((root / "labels.json").read_text()), RecordedCandidate())
    assert result["recorded"]["returned"] == result["recorded"]["correct"] == 1
    assert result["recorded"]["provider"] == "test"
    assert "model_sha256" not in result["recorded"]


def test_short_loss_can_exceed_initial_capital():
    ds = load_dataset(FIXTURES / "nse")
    row = dict(json.loads(SPEC.read_text())["sleeves"][1]["borrow"][0])
    row["max_units"] = 10000
    schedule = BorrowSchedule(ds.meta["instrument"], [row])
    cfg = Assumptions(cash=D(1000), fee_bps=D(0), slippage_bps=D(0), participation=D(1))
    engine = PaperEngine(ds.meta, "short_hold", cfg, schedule)
    engine.observe(ds.bars[0])
    engine.observe(ds.bars[1])
    engine.observe(replace(ds.bars[2], open=D(400), close=D(400), high=D(400), low=D(400)))
    assert engine.ledger[-1]["insolvent"]
    assert engine.ledger[-1]["drawdown"] > 1
    assert engine.ledger[-1]["gross_exposure"] is None
