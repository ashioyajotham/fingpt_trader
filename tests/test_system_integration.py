import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from main import TradingSystem  # Changed from fingpt_trader import TradingSystem

@pytest.fixture
async def trading_system():
    system = TradingSystem(config_path="config/test_trading.yaml")
    await system.initialize()
    yield system
    await system.shutdown()

@pytest.mark.asyncio
async def test_full_trading_cycle(trading_system):
    # Test market data acquisition
    market_data = await trading_system.get_market_data()
    assert 'prices' in market_data
    assert 'volume' in market_data
    assert 'sentiment' in market_data
    
    # Test signal generation
    signals = trading_system.detect_inefficiencies(market_data)
    assert isinstance(signals, dict)
    assert 'confidence' in signals
    assert 'magnitude' in signals
    
    # Test trade generation
    trades = trading_system.generate_trades(signals)
    assert isinstance(trades, list)
    for trade in trades:
        assert 'symbol' in trade
        assert 'size' in trade
        assert 'direction' in trade

@pytest.mark.asyncio
async def test_risk_monitoring(trading_system):
    # Setup mock portfolio
    await trading_system.set_portfolio({
        'positions': {'AAPL': 100, 'MSFT': 150},
        'cash': 100000
    })
    
    # Test risk metrics calculation
    risk_metrics = trading_system.update_risk_metrics()
    assert 'var' in risk_metrics
    assert 'sharpe' in risk_metrics
    assert 'max_drawdown' in risk_metrics

@pytest.mark.asyncio
async def test_error_handling(trading_system):
    with pytest.raises(Exception):
        await trading_system.execute_trades(None)
    
    with pytest.raises(ValueError):
        await trading_system.get_market_data(symbol="INVALID")
