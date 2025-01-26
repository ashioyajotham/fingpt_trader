import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from strategies.base_strategy import BaseStrategy
from strategies.inefficiency.inefficiency_strategy import InefficiencyStrategy
from strategies.sentiment.sentiment_strategy import SentimentStrategy


class HybridStrategy(BaseStrategy):
    """
    Combined trading strategy using sentiment and technical analysis.
    
    Strategy Components:
    - Sentiment Analysis: News and social media sentiment
    - Technical Analysis: Price patterns and indicators
    - Risk Management: Position sizing and stop-loss
    
    Configuration:
        weights: Strategy component weights
        thresholds: Signal generation thresholds
        risk_limits: Position and portfolio limits
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.signal_history = []
        self.max_history = self.config.get("max_history", 1000)
        self.weights = self.config.get("weights", {"sentiment": 0.6, "technical": 0.4})
        self.signal_threshold = self.config.get("signal_threshold", 0.5)

        # Initialize sub-strategies with default configs
        self.strategies = {
            "sentiment": SentimentStrategy(self.config.get("sentiment_config", {})),
            "technical": InefficiencyStrategy(self.config.get("technical_config", {})),
        }

        # Get weights from config or use defaults
        weights = self.config.get("weights", {})
        self.sentiment_weight = weights.get("sentiment", 0.6)
        self.technical_weight = weights.get("technical", 0.4)

        self.signal_threshold = self.config.get("signal_threshold", 0.5)
        self.market_data = {}
        self.signals = []

    async def generate_signals(self) -> List[Dict]:
        sentiment_signals = await self.strategies["sentiment"].generate_signals()
        technical_signals = await self.strategies["technical"].generate_signals()

        combined = self._combine_signals(sentiment_signals, technical_signals)
        self.signal_history.append(combined)

        # Maintain history size
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history :]

        return combined

    def _combine_signals(
        self, sentiment_signals: List[Dict], technical_signals: List[Dict]
    ) -> List[Dict]:
        combined = []
        for symbol in set([s["symbol"] for s in sentiment_signals + technical_signals]):
            sent_score = next(
                (s["strength"] for s in sentiment_signals if s["symbol"] == symbol), 0
            )
            tech_score = next(
                (s["strength"] for s in technical_signals if s["symbol"] == symbol), 0
            )

            total_score = (
                sent_score * self.weights["sentiment"]
                + tech_score * self.weights["technical"]
            )

            if abs(total_score) > self.signal_threshold:
                combined.append(
                    {
                        "symbol": symbol,
                        "strength": total_score,
                        "direction": 1 if total_score > 0 else -1,
                        "timestamp": datetime.now(),
                        "type": "hybrid",
                    }
                )

        return combined

    async def _update_weights(self) -> None:
        """Update strategy weights based on performance"""
        if len(self.signal_history) < 10:
            return

        # Calculate strategy returns
        returns = self._calculate_strategy_returns()

        # Update correlation matrix
        self.correlation_matrix = np.corrcoef(returns)

        # Adjust weights using return/risk ratio
        total_return = np.sum(returns, axis=1)
        total_risk = np.sqrt(np.diag(returns @ self.correlation_matrix @ returns.T))

        new_weights = total_return / total_risk
        new_weights = new_weights / np.sum(new_weights)

        for i, strategy in enumerate(self.strategies):
            self.weights[strategy] = new_weights[i]

    async def process_market_data(self, data: Dict) -> None:
        """Process market data updates"""
        self.market_data.update(data)
        await self._generate_signals()

    async def on_trade(self, trade: Dict) -> None:
        """Handle trade updates"""
        symbol = trade.get("symbol")
        if symbol in self.positions:
            self.positions[symbol].update(trade)

    async def _generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        for symbol, data in self.market_data.items():
            signal = self._analyze_symbol(symbol, data)
            if signal:
                signals.append(signal)
        return signals

    def _analyze_symbol(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Analyze single symbol"""
        sentiment_score = data.get("sentiment", 0)
        technical_score = data.get("technical", 0)

        combined_score = (
            self.sentiment_weight * sentiment_score
            + self.technical_weight * technical_score
        )

        if abs(combined_score) > self.config.get("signal_threshold", 0.5):
            return {
                "symbol": symbol,
                "direction": np.sign(combined_score),
                "strength": abs(combined_score),
                "type": "hybrid",
            }
        return None
