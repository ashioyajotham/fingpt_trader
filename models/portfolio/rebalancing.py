from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from models.portfolio.risk import MarketRegime, MarketRegimeDetector, CircuitBreaker

import logging
logger = logging.getLogger(__name__)

class Portfolio:
    """Portfolio management class for tracking positions and performance"""
    
    def __init__(self, initial_balance: float):
        self.cash = initial_balance
        self.positions = {}
        self.position_entries = {}  # Track entry prices
        self.trades = []
        self.total_trades = 0
        self.portfolio_history = []
        self.last_prices = {}  # Track last known prices

    def initialize_with_positions(self, positions: Dict[str, float], prices: Dict[str, float]):
        """Initialize portfolio with positions and their current prices"""
        for symbol, position_size in positions.items():
            # Get price for this symbol
            price = prices.get(symbol, 0)
            if price <= 0:
                logger.warning(f"No valid price for {symbol}, using 0")
                
            # Add position with entry price tracking
            self.positions[symbol] = position_size
            self.position_entries[symbol] = price  # Store entry price
            
            # Update last known price
            self.last_prices[symbol] = price
        
        logger.info(f"Portfolio initialized with positions: {positions}")

    def update_prices(self, market_data: Dict[str, Dict[str, float]]) -> None:
        """Update last known prices for all assets
        
        Args:
            market_data: Dict of symbol -> {price: float, ...}
        """
        for symbol, data in market_data.items():
            self.last_prices[symbol] = data['price']

    async def initialize(self, config: dict = None) -> None:
        """Initialize portfolio with configuration"""
        if config is None:
            config = {}
        # Initialize any additional settings from config
        self.position_limits = config.get('position_limits', {})
        self.risk_limits = config.get('risk_limits', {})
        logger.info("Portfolio initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup portfolio resources"""
        # Clear positions and state
        self.positions.clear()
        self.position_entries.clear()
        self.last_prices.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        logger.info("Portfolio cleaned up successfully")

    async def buy(self, symbol: str, size: float, price: float):
        cost = size * price
        if cost <= self.cash:
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            self.trades.append({
                'type': 'BUY',
                'symbol': symbol,
                'size': size,
                'price': price,
                'timestamp': pd.Timestamp.now()
            })
            self.total_trades += 1

    async def sell(self, symbol: str, size: float, price: float):
        if symbol in self.positions and self.positions[symbol] >= size:
            self.cash += size * price
            self.positions[symbol] -= size
            if self.positions[symbol] == 0:
                del self.positions[symbol]
            self.trades.append({
                'type': 'SELL',
                'symbol': symbol,
                'size': size,
                'price': price,
                'timestamp': pd.Timestamp.now()
            })
            self.total_trades += 1

    async def add_position(self, symbol: str, size: float, price: float, cost: float = None):
        """Add or increase a position"""
        try:
            # Initialize position_entries if it doesn't exist
            if not hasattr(self, 'position_entries'):
                self.position_entries = {}
                
            # Calculate cost if not provided
            if cost is None:
                cost = size * price
                
            # Add to position
            current_position = self.positions.get(symbol, 0)
            self.positions[symbol] = current_position + size
            
            # Track entry price using weighted average
            if symbol not in self.position_entries:
                # First position - simple entry price
                self.position_entries[symbol] = price
            else:
                # Calculate weighted average for entry price
                old_size = current_position
                old_price = self.position_entries[symbol]
                total_size = old_size + size
                
                # Weighted average calculation for entry price
                if total_size > 0:  # Prevent division by zero
                    self.position_entries[symbol] = ((old_size * old_price) + (size * price)) / total_size
            
            # Reduce cash
            self.cash -= cost
            
            # Update last known price
            if not hasattr(self, 'last_prices'):
                self.last_prices = {}
            self.last_prices[symbol] = price
            
            # Log for debugging
            logger.info(f"Position added: {size} {symbol} @ {price} (Entry price: {self.position_entries[symbol]})")
            
            # Record portfolio state
            self._record_portfolio_state()
            
            return True
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False

    async def reduce_position(self, symbol: str, size: float, price: float):
        """Reduce or close an existing position"""
        try:
            # Check if position exists and has sufficient size
            if symbol not in self.positions or self.positions[symbol] < size:
                logger.warning(f"Insufficient position: {self.positions.get(symbol, 0)} < {size}")
                return False
            
            # Store original position size and entry price for P&L calculation
            original_size = self.positions[symbol]
            entry_price = self.position_entries.get(symbol, 0)
            
            # Update position
            self.positions[symbol] -= size
            
            # Only remove entry price if position is fully closed
            if self.positions[symbol] <= 0:
                if symbol in self.position_entries:
                    del self.position_entries[symbol]
                if symbol in self.positions:
                    del self.positions[symbol]
            
            # Calculate P&L for this trade
            pnl = (price - entry_price) * size
            
            # Record the trade with P&L
            self.trades.append({
                'type': 'SELL',
                'symbol': symbol,
                'size': size,
                'price': price, 
                'proceeds': size * price,
                'entry_price': entry_price,
                'pnl': pnl,
                'timestamp': datetime.now()
            })
            
            # Add proceeds to cash
            self.cash += size * price
            
            # Update last known price
            self.last_prices[symbol] = price
            
            # Record portfolio state
            self._record_portfolio_state()
            
            logger.info(f"Position reduced: {size} {symbol} @ {price} (Proceeds: {size * price:.2f}, P&L: {pnl:.2f})")
            return True
        except Exception as e:
            logger.error(f"Error reducing position: {e}")
            return False

    def _record_portfolio_state(self):
        """Record current portfolio state for historical tracking"""
        # Calculate total value
        total_value = self.cash
        for symbol, size in self.positions.items():
            price = self.last_prices.get(symbol, 0)
            position_value = size * price
            total_value += position_value
        
        # Record state
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'cash': self.cash,
            'positions': self.positions.copy(),
            'total_value': total_value
        })
        
        # Limit history size
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0)

    def get_position_value(self, symbol: str) -> float:
        if symbol not in self.last_prices:
            logger.warning(f"No price data for {symbol}")
            return 0.0
        return self.positions.get(symbol, 0) * self.last_prices[symbol]

    def total_value(self) -> float:
        if not self.last_prices:
            logger.warning("No price data available for portfolio valuation")
            return self.cash
        return self.cash + sum(
            pos * self.last_prices.get(sym, 0)
            for sym, pos in self.positions.items()
        )

    def get_position_size(self, cash: float, price: float) -> float:
        # Use a portion of available cash (e.g., 95% to leave room for fees)
        return (cash * 0.95) / price

    def calculate_sharpe_ratio(self) -> float:
        if not self.portfolio_history:
            return 0.0
        returns = pd.Series([x['total_value'] for x in self.portfolio_history]).pct_change()
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def calculate_max_drawdown(self) -> float:
        if not self.portfolio_history:
            return 0.0
        values = pd.Series([x['total_value'] for x in self.portfolio_history])
        peaks = values.expanding(min_periods=1).max()
        drawdowns = (values - peaks) / peaks
        return abs(drawdowns.min())

    def calculate_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        profitable_trades = sum(1 for t in self.trades if 
            (t['type'] == 'SELL' and t['price'] > t.get('entry_price', 0)) or
            (t['type'] == 'BUY' and t['price'] < t.get('entry_price', 0)))
        return profitable_trades / len(self.trades)

    def calculate_profit_factor(self) -> float:
        gains = sum(t['price'] * t['size'] for t in self.trades 
                   if t['type'] == 'SELL' and t['price'] > t.get('entry_price', 0))
        losses = sum(t['price'] * t['size'] for t in self.trades 
                    if t['type'] == 'SELL' and t['price'] < t.get('entry_price', 0))
        return gains / losses if losses != 0 else 0.0


class PortfolioRebalancer:
    def __init__(self, config: Dict):
        self.config = config
        self.regime_detector = MarketRegimeDetector()
        self.circuit_breaker = CircuitBreaker(config.get('risk', {}).get('thresholds', {}))
        
        # Regime-based rebalancing thresholds
        self.regime_thresholds = {
            MarketRegime.NORMAL: 0.05,        # 5% deviation trigger
            MarketRegime.HIGH_VOL: 0.08,      # 8% in high volatility
            MarketRegime.STRESS: 0.10,        # 10% in stress
            MarketRegime.CRISIS: 1.0,         # No rebalancing in crisis
            MarketRegime.LOW_LIQUIDITY: 0.15  # 15% in low liquidity
        }

    async def check_rebalance_needed(
        self, 
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_data: Dict
    ) -> bool:
        """Check if rebalancing is needed based on market regime"""
        # Check circuit breaker first
        if self.circuit_breaker.check_conditions(market_data):
            return False  # Don't rebalance if circuit breaker triggered
            
        # Detect current market regime
        current_regime = self.regime_detector.detect_regime(market_data)
        threshold = self.regime_thresholds[current_regime]
        
        # Calculate maximum deviation
        max_deviation = 0.0
        for asset in target_weights:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)
            
        return max_deviation > threshold

    async def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        market_data: Dict
    ) -> Optional[Dict[str, float]]:
        """Calculate required trades for rebalancing"""
        try:
            if not await self.check_rebalance_needed(
                current_positions, target_weights, market_data
            ):
                return None
                
            # Calculate trades considering market impact
            trades = {}
            total_value = sum(current_positions.values())
            
            for asset, target in target_weights.items():
                current = current_positions.get(asset, 0.0)
                target_value = total_value * target
                trade_size = target_value - current
                
                # Apply market regime-based size limits
                regime = self.regime_detector.detect_regime(market_data)
                if regime != MarketRegime.NORMAL:
                    trade_size *= self.regime_thresholds[regime]
                    
                trades[asset] = trade_size
                
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalance trades: {str(e)}")
            return None
