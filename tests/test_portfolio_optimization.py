import pytest
import numpy as np

import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.portfolio.optimization import PortfolioOptimizer
from models.portfolio.risk import RiskManager


@pytest.fixture
def returns_data():
    # Generate sample returns data for 5 assets
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, (100, 5))

@pytest.fixture
def optimizer():
    return PortfolioOptimizer({
        'risk_free_rate': 0.02,
        'constraints': {
            'min_weight': 0.0,
            'max_weight': 0.3,
            'min_positions': 5
        }
    })

def test_portfolio_optimization(optimizer, returns_data):
    weights = optimizer.optimize(returns_data)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == returns_data.shape[1]
    assert np.isclose(np.sum(weights), 1.0)
    assert all(0 <= w <= 0.3 for w in weights)

def test_risk_management():
    risk_manager = RiskManager(max_drawdown=0.1, var_limit=0.02)
    
    portfolio = {
        'positions': np.array([0.2, 0.3, 0.5]),
        'values': np.array([100000, 150000, 250000])
    }
    
    risk_metrics = risk_manager.calculate_risk_metrics(portfolio)
    assert 'var' in risk_metrics
    assert 'current_drawdown' in risk_metrics
    assert risk_metrics['var'] > 0
