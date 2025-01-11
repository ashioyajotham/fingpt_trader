from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class RiskMetric(Enum):
    VOLATILITY = "volatility"
    VALUE_AT_RISK = "var"
    EXPECTED_SHORTFALL = "es"


@dataclass
class RiskBudget:
    symbol: str
    max_position: float
    max_risk: float
    stop_loss: float
    take_profit: float


class RiskAllocator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.total_risk_budget = self.config.get("total_risk_budget", 1.0)
        self.max_position_size = self.config.get("max_position_size", 0.2)
        self.risk_metric = RiskMetric(self.config.get("risk_metric", "volatility"))
        self.correlation_matrix = np.eye(1)
        self.position_history = []

    def allocate_risk(
        self, signals: List[Dict], portfolio_value: float, market_data: Dict
    ) -> Dict[str, RiskBudget]:
        """Calculate risk allocations for signals"""
        # Calculate individual position risks
        position_risks = self._calculate_position_risks(signals, market_data)

        # Apply Kelly criterion for position sizing
        kelly_fractions = self._calculate_kelly_fractions(signals)

        # Generate risk budgets
        risk_budgets = {}
        for signal in signals:
            symbol = signal["symbol"]
            position_risk = position_risks.get(symbol, 0)
            kelly_fraction = kelly_fractions.get(symbol, 0)

            max_position = min(
                kelly_fraction * portfolio_value,
                self.max_position_size * portfolio_value,
            )

            risk_budgets[symbol] = RiskBudget(
                symbol=symbol,
                max_position=max_position,
                max_risk=position_risk * self.total_risk_budget,
                stop_loss=self._calculate_stop_loss(signal),
                take_profit=self._calculate_take_profit(signal),
            )

        return risk_budgets

    def _calculate_position_risks(
        self, signals: List[Dict], market_data: Dict
    ) -> Dict[str, float]:
        """Calculate risk metrics for each position"""
        risks = {}
        for signal in signals:
            symbol = signal["symbol"]
            if symbol not in market_data:
                continue

            prices = market_data[symbol]["close"]
            if len(prices) < 2:
                continue

            if self.risk_metric == RiskMetric.VOLATILITY:
                risks[symbol] = np.std(np.diff(prices) / prices[:-1])
            elif self.risk_metric == RiskMetric.VALUE_AT_RISK:
                risks[symbol] = self._calculate_var(prices)
            else:
                risks[symbol] = self._calculate_es(prices)

        return risks

    def _calculate_kelly_fractions(self, signals: List[Dict]) -> Dict[str, float]:
        """Calculate Kelly criterion position sizes"""
        kelly_fractions = {}
        for signal in signals:
            win_prob = signal.get("confidence", 0.5)
            win_loss_ratio = signal.get("strength", 1.0)

            kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly_fractions[signal["symbol"]] = max(0, kelly)

        return kelly_fractions

    def _calculate_stop_loss(self, signal: Dict) -> float:
        """Calculate stop-loss level"""
        base_stop = 0.02  # 2% default stop-loss
        return base_stop * (1 / signal.get("confidence", 1.0))

    def _calculate_take_profit(self, signal: Dict) -> float:
        """Calculate take-profit level"""
        return self._calculate_stop_loss(signal) * signal.get("strength", 2.0)
