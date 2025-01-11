import pytest
from dataclasses import dataclass
from typing import Dict

import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.trading.robo_service import RoboAdvisor


@dataclass
class MockClientProfile:
    risk_score: float
    investment_horizon: int
    constraints: Dict
    tax_rate: float
    esg_preferences: Dict

@pytest.fixture
def client_profile():
    return MockClientProfile(
        risk_score=7.0,
        investment_horizon=10,
        constraints={'max_stock': 0.8, 'min_bonds': 0.2},
        tax_rate=0.25,
        esg_preferences={'environmental': 0.4, 'social': 0.3, 'governance': 0.3}
    )

@pytest.fixture
def robo_advisor():
    return RoboAdvisor(config={
        'rebalance_threshold': 0.05,
        'tax_impact_factor': 0.3,
        'base_threshold': 0.02
    })

def test_portfolio_construction(robo_advisor, client_profile):
    portfolio = robo_advisor.construct_portfolio(client_profile)
    assert isinstance(portfolio, dict)
    assert 'weights' in portfolio
    assert 'implementation' in portfolio
    assert sum(portfolio['weights'].values()) == pytest.approx(1.0)

def test_tax_loss_harvesting(robo_advisor):
    current_portfolio = {
        'AAPL': {'cost_basis': 150, 'current_price': 140, 'quantity': 100},
        'MSFT': {'cost_basis': 200, 'current_price': 220, 'quantity': 50}
    }
    
    trades = robo_advisor.harvest_tax_losses(
        portfolio=current_portfolio,
        tax_rate=0.25,
        wash_sale_window=30
    )
    
    assert isinstance(trades, list)
    for trade in trades:
        assert 'symbol' in trade
        assert 'quantity' in trade
        assert 'action' in trade

def test_esg_optimization(robo_advisor):
    portfolio = {
        'AAPL': 0.3,
        'MSFT': 0.3,
        'GOOGL': 0.4
    }
    
    esg_scores = {
        'AAPL': 0.8,
        'MSFT': 0.9,
        'GOOGL': 0.7
    }
    
    optimized = robo_advisor.optimize_esg_portfolio(
        weights=portfolio,
        esg_scores=esg_scores,
        min_esg_score=0.75
    )
    
    assert isinstance(optimized, dict)
    assert sum(optimized.values()) == pytest.approx(1.0)
    
    # Calculate portfolio ESG score
    portfolio_esg = sum(w * esg_scores[s] for s, w in optimized.items())
    assert portfolio_esg >= 0.75
