import shutil
from datetime import date, timedelta
from decimal import Decimal

import pytest

from asof.cli import get_parser, run as run_cli
from asof.dividends import CORPUS, actions, promote_filing
from asof.factor_backtest import FactorBacktestConfig, _target_weights, backtest_factor
from asof.factors import load_universe
from asof.financials import promote_financials
from asof.nse_pricelist import import_nse_universe


def _build_multi_month_universe(tmp_path):
    """Build a deterministic 4-ticker 45-session NSE universe spanning 3 calendar months."""
    csv_path = tmp_path / "nse_45d.csv"
    lines = ["Date,Code,Name,Day Low,Day High,Day Price,Previous,Volume"]
    specs = [
        ("SCOM", "Safaricom Plc", 20.0, 0.012, 5_000_000),  # strong uptrend
        ("EQTY", "Equity Group", 45.0, 0.006, 2_000_000),   # moderate uptrend
        ("KCB", "KCB Group", 35.0, 0.000, 1_000_000),       # flat
        ("KPLC", "Kenya Power", 10.0, -0.010, 500_000),     # downtrend
    ]
    days = []
    cur = date(2025, 1, 2)
    while len(days) < 45:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)

    for code, name, p0, drift, vol in specs:
        prev = p0
        for idx, d in enumerate(days):
            price = round(p0 * ((1.0 + drift) ** idx), 2)
            lo = round(min(price, prev) * 0.99, 2)
            hi = round(max(price, prev) * 1.01, 2)
            lines.append(f"{d.isoformat()},{code},{name},{lo},{hi},{price},{prev},{vol}")
            prev = price
    csv_path.write_text("\n".join(lines) + "\n")
    uni_dir = tmp_path / "universe"
    import_nse_universe(csv_path, uni_dir)
    return uni_dir, days


def test_target_weights_respects_max_weight_cap():
    w_eq = _target_weights(["SCOM", "EQTY"], "equal", Decimal("0.35"))
    assert w_eq == {"SCOM": Decimal("0.35"), "EQTY": Decimal("0.35")}
    w_rank = _target_weights(["SCOM", "EQTY", "KCB", "KPLC"], "rank", Decimal("0.35"))
    assert max(w_rank.values()) <= Decimal("0.35")
    assert sum(w_rank.values()) == pytest.approx(Decimal("1"))


def test_factor_backtest_top_vs_bottom_direction_and_benchmark_attribution(tmp_path):
    uni_dir, _ = _build_multi_month_universe(tmp_path)
    universe = load_universe(uni_dir)

    top_rep = backtest_factor(
        universe,
        FactorBacktestConfig(
            factor="momentum_20d", direction="top", top_n=2, rebalance="weekly",
            max_weight=Decimal("0.5"), cash=Decimal("1000000"),
        ),
    )
    bot_rep = backtest_factor(
        universe,
        FactorBacktestConfig(
            factor="momentum_20d", direction="bottom", top_n=2, rebalance="weekly",
            max_weight=Decimal("0.5"), cash=Decimal("1000000"),
        ),
    )

    assert top_rep["mode"] == "factor_backtest"
    assert top_rep["rebalance_count"] >= 3
    assert top_rep["trades_count"] >= 2
    # In a monotonic drift universe, long top-momentum must strongly beat long bottom-momentum and EW benchmark.
    assert top_rep["net_return"] > 0
    assert bot_rep["net_return"] < 0
    assert top_rep["benchmark_attribution"]["active_return"] > 0
    assert top_rep["final_holdings"]
    assert all(h["symbol"] in {"SCOM", "EQTY"} for h in top_rep["final_holdings"])
    assert top_rep["total_fees"] > 0


def test_factor_backtest_zero_lookahead_and_cli_execution(tmp_path):
    uni_dir, days = _build_multi_month_universe(tmp_path)
    cat_dir = tmp_path / "catalog"
    # Disclose EPS on days[10]; available only from days[11] onward under Nairobi midnight clock.
    promote_financials({
        "symbol": "SCOM", "source": "https://example.test/scom.pdf", "pdf_sha256": "abc",
        "financials": {"eps": {"value": "3.0", "raw_value": "3.0", "currency": "KES",
                               "unit": "KES/share", "scale": "1", "period": "2024",
                               "citations": [{"page": 1, "quote": "EPS 3.00"}]}},
    }, days[10].isoformat(), cat_dir)

    rep = run_cli(get_parser().parse_args([
        "factor-backtest",
        "--universe", str(uni_dir),
        "--catalog", str(cat_dir),
        "--factor", "earnings_yield",
        "--direction", "top",
        "--top-n", "1",
        "--rebalance", "daily",
        "--max-weight", "0.5",
        "--fee-bps", "20",
    ]))
    # First rebalance signal cannot occur on or before days[10] (disclosure date).
    first_signal = rep["rebalances"][0]["signal_session"]
    assert first_signal >= days[11].isoformat()
    # First position entry cannot occur on the signal session itself (must fill on next trading session).
    by_sess = {row["session"]: row for row in rep["ledger"]}
    assert by_sess[first_signal]["positions_count"] == 0
    next_sess = days[days.index(date.fromisoformat(first_signal)) + 1].isoformat()
    assert by_sess[next_sess]["positions_count"] == 1


def test_promote_filing_into_seeded_corpus_inherits_split_and_preserves_adjudications(tmp_path):
    local_cat = tmp_path / "seeded-catalog"
    shutil.copytree(CORPUS, local_cat)
    report = {
        "symbol": "SCBK",
        "source": "https://av.sc.com/ke/content/docs/scbk-extra.pdf",
        "pdf_sha256": "deadbeef",
        "events": [{
            "event_id": "event-01",
            "fields": {
                "status": {"value": "proposed", "citations": [{"page": 2, "quote": "final dividend of KShs 37.00"}]},
                "amount": {"value": "37.00", "citations": [{"page": 2, "quote": "final dividend of KShs 37.00"}]},
                "currency": {"value": "KES", "citations": [{"page": 2, "quote": "final dividend of KShs 37.00"}]},
                "distribution_type": {"value": "final", "citations": [{"page": 2, "quote": "final dividend of KShs 37.00"}]},
                "period": {"value": "2024", "citations": [{"page": 2, "quote": "final dividend of KShs 37.00"}]},
            },
        }],
    }
    res = promote_filing(report, "2025-03-19", local_cat)
    assert res["total_notices"] == 43
    assert not (local_cat / "adjudication.json").exists()
    out = actions("2025-03-20", symbol="SCBK", root=local_cat)
    assert any(r["event"] == "SCBK-2024-final" for r in out["rows"])
