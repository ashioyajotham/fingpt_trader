from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from models.portfolio.risk import MarketRegime, MarketRegimeDetector, CircuitBreaker

import logging
logger = logging.getLogger(__name__)

class Portfolio:
    def __init__(self):
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.prices: Dict[str, float] = {}  # symbol -> current price
        self.target_weights: Dict[str, float] = {}  # symbol -> target weight

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices"""
        self.prices = prices

    def get_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {symbol: 0.0 for symbol in self.positions}

        weights = {}
        for symbol in self.positions:
            position_value = self.positions[symbol] * self.prices[symbol]
            weights[symbol] = position_value / total_value
        return weights

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(self.positions[s] * self.prices[s] for s in self.positions)

    def equal_weight_rebalance(self) -> List[Tuple[str, float]]:
        """Rebalance to equal weights across all assets"""
        n_assets = len(self.positions)
        if n_assets == 0:
            return []

        target_weight = 1.0 / n_assets
        self.target_weights = {symbol: target_weight for symbol in self.positions}
        return self._generate_rebalance_trades()

    def target_weight_rebalance(
        self, target_weights: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Rebalance to specified target weights"""
        if abs(sum(target_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Target weights must sum to 1.0")

        self.target_weights = target_weights
        return self._generate_rebalance_trades()

    def threshold_rebalance(self, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """Rebalance only if any position deviated more than threshold"""
        current_weights = self.get_current_weights()

        for symbol, weight in current_weights.items():
            if abs(weight - self.target_weights[symbol]) > threshold:
                return self._generate_rebalance_trades()
        return []

    def _generate_rebalance_trades(self) -> List[Tuple[str, float]]:
        """Generate trades to achieve target weights"""
        portfolio_value = self.get_portfolio_value()
        trades = []

        for symbol in self.positions:
            target_value = portfolio_value * self.target_weights[symbol]
            current_value = self.positions[symbol] * self.prices[symbol]
            trade_value = target_value - current_value

            if abs(trade_value) > 1e-6:  # Ignore tiny trades
                trade_quantity = trade_value / self.prices[symbol]
                trades.append((symbol, trade_quantity))

        return trades


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
