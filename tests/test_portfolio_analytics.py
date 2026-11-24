import json
from decimal import Decimal

import pytest

from asof.portfolio import portfolio
from asof.risk import benchmark_attribution, concentration_metrics, sortino_ratio


def test_sortino_concentration_and_benchmark_attribution():
    assert sortino_ratio([100, 102, 105]) is None  # zero downside variance
    assert sortino_ratio([100, 99]) is None
    s = sortino_ratio([100, 103, 98, 104, 109])
    assert s is not None and s > 0

    c = concentration_metrics({"A": 0.6, "B": 0.4})
    assert c["max_weight"] == pytest.approx(0.6)
    assert c["hhi"] == pytest.approx(0.52)
    assert c["effective_positions"] == pytest.approx(1.0 / 0.52)
    assert concentration_metrics({})["effective_positions"] == 0.0

    attr = benchmark_attribution([100, 102, 101, 106, 110], [100, 101, 100.5, 103, 105])
    assert attr is not None
    assert attr["active_return"] == pytest.approx(0.05)
    assert attr["tracking_error"] is not None and attr["tracking_error"] > 0
    assert attr["information_ratio"] is not None and attr["information_ratio"] > 0
    assert attr["beta"] is not None and attr["beta"] > 1.0


def test_portfolio_rebalance_concentration_cap_and_benchmark_attribution(tmp_path):
    from pathlib import Path

    base = json.loads(Path("examples/kes-portfolio.json").read_text())
    spec = {
        **base,
        "rebalance": "daily",
        "max_single_weight": "0.60",
        "benchmark": "binance",
        "sleeves": [
            {"dataset": "nse", "policy": "cash", "weight": "0.60"},
            {"dataset": "binance", "policy": "buy_hold", "weight": "0.40"},
        ],
    }
    spec_path = tmp_path / "rebalanced.json"
    spec_path.write_text(json.dumps(spec))
    result = portfolio(spec_path)
    assert result["concentration"]["max_weight"] <= 0.70
    assert result["concentration"]["effective_positions"] > 1.5
    assert set(result["liquidity"]) == {"XNAI:DEMO-001", "BINANCE:BTCUSDT"}
    assert result["benchmark_attribution"] is not None
    assert "active_return" in result["benchmark_attribution"]
    # Cash transfers during rebalance must sum to zero in base currency.
    for ev in result["rebalances"]:
        assert abs(sum(ev["transfers"].values(), Decimal(0))) < Decimal("1e-18")

    # Exceeding max_single_weight must fail fast.
    bad = {**spec, "max_single_weight": "0.50"}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="max_single_weight"):
        portfolio(bad_path)
