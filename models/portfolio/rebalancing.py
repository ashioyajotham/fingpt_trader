from typing import Dict, List, Tuple

import numpy as np


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
