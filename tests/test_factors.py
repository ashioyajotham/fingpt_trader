import json
from datetime import date, timedelta

import pytest

from asof.cli import get_parser, run as run_cli
from asof.factors import compute_factor_panel, evaluate_factors, load_universe, spearman_rank_ic, universe_snapshot
from asof.financials import promote_financials
from asof.nse_pricelist import import_nse_universe


def _build_universe(tmp_path):
    """Build a deterministic 4-ticker 30-session NSE universe with monotonically ordered trends."""
    csv_path = tmp_path / "nse_30d.csv"
    lines = ["Date,Code,Name,Day Low,Day High,Day Price,Previous,Volume"]
    # 4 tickers with distinct daily drifts and volumes so momentum & liquidity Rank-IC are deterministic.
    specs = [
        ("SCOM", "Safaricom Plc", 20.0, 0.015, 5000000),   # strongest uptrend + highest ADV
        ("EQTY", "Equity Group", 45.0, 0.008, 2000000),
        ("KCB", "KCB Group", 35.0, 0.001, 1000000),
        ("KPLC", "Kenya Power", 5.0, -0.010, 200000),      # downtrend + lowest ADV
    ]
    start = date(2025, 1, 2)
    days = []
    cur = start
    while len(days) < 30:
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


def test_spearman_rank_ic_exact_and_tied_ranks():
    assert spearman_rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rank_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_rank_ic([1, 1, 2, 3], [10, 10, 20, 30]) == pytest.approx(1.0)
    assert spearman_rank_ic([1, 2], [3, 4]) is None
    assert spearman_rank_ic([5, 5, 5], [1, 2, 3]) is None


def test_cross_sectional_factors_rank_ic_and_zero_lookahead(tmp_path):
    uni_dir, days = _build_universe(tmp_path)
    cat_dir = tmp_path / "catalog"
    # Promote EPS for SCOM on day 15 (visible starting day 16 under Nairobi midnight clock).
    eps_date = days[14].isoformat()
    promote_financials({
        "symbol": "SCOM", "source": "https://example.test/scom.pdf", "pdf_sha256": "abc",
        "financials": {"eps": {"value": "2.5", "raw_value": "2.5", "currency": "KES",
                               "unit": "KES/share", "scale": "1", "period": "2025",
                               "citations": [{"page": 1, "quote": "EPS 2.50"}]}},
    }, eps_date, cat_dir)

    universe = load_universe(uni_dir)
    panel, _ = compute_factor_panel(universe, catalog_dir=cat_dir)
    # Zero look-ahead: EPS disclosed on days[14] must NOT appear on days[14], MUST appear on days[15].
    assert "earnings_yield" not in panel[days[14].isoformat()]["SCOM"]
    assert panel[days[15].isoformat()]["SCOM"]["earnings_yield"] > 0

    # Evaluate factors via CLI.
    report = run_cli(get_parser().parse_args([
        "factors", "--universe", str(uni_dir), "--catalog", str(cat_dir), "--horizons", "1,5",
    ]))
    assert report["tickers"] == ["EQTY", "KCB", "KPLC", "SCOM"]
    # Persistent monotonic drift -> momentum_20d has Rank-IC == +1.0 and positive quantile spread.
    mom_1d = report["factors"]["momentum_20d"]["1"]
    assert mom_1d["evaluated_sessions"] > 0
    assert mom_1d["mean_ic"] == pytest.approx(1.0)
    assert mom_1d["hit_rate"] == pytest.approx(1.0)
    assert mom_1d["mean_spread"] > 0

    # Liquidity filter excludes KPLC (lowest KES ADV ~1M KES vs others >35M KES).
    filtered = evaluate_factors(universe, horizons=(1,), min_adv=5_000_000)
    assert filtered["factors"]["liquidity_adv20"]["1"]["avg_universe_size"] == pytest.approx(3.0)

    # Universe snapshot returns all 4 tickers with earnings_yield populated for SCOM.
    snap = universe_snapshot(universe, catalog_dir=cat_dir)
    by_sym = {r["symbol"]: r for r in snap["rows"]}
    assert snap["tickers_count"] == 4
    assert by_sym["SCOM"]["earnings_yield"] is not None
    assert by_sym["SCOM"]["stale"] is False
