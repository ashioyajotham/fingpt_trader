import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from .patterns import PatternDetector


@dataclass
class InefficencySignal:
    type: str  # Type of inefficiency
    confidence: float  # Signal confidence [0-1]
    magnitude: float  # Expected price impact
    horizon: str  # Expected time horizon
    metadata: Dict  # Additional signal info


class MarketInefficencyDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.pattern_detector = PatternDetector(config.get("pattern_config", {}))
        self.lookback_periods = config.get(
            "lookback_periods", {"short": 5, "medium": 20, "long": 60}
        )

    def detect_inefficiencies(
        self,
        prices: pd.DataFrame,
        volume: pd.Series,
        sentiment: Optional[pd.Series] = None,
    ) -> List[InefficencySignal]:
        """Detect market inefficiencies from multiple sources"""
        signals = []

        # Technical pattern-based inefficiencies
        pattern_signals = self._detect_pattern_inefficiencies(prices, volume)
        signals.extend(pattern_signals)

        # Behavioral inefficiencies
        behavior_signals = self._detect_behavioral_inefficiencies(
            prices, volume, sentiment
        )
        signals.extend(behavior_signals)

        # Liquidity-based inefficiencies
        liquidity_signals = self._detect_liquidity_inefficiencies(prices, volume)
        signals.extend(liquidity_signals)

        # Filter and sort signals by confidence
        return sorted(
            [
                s
                for s in signals
                if s.confidence >= self.config.get("min_confidence", 0.6)
            ],
            key=lambda x: x.confidence,
            reverse=True,
        )

    def _detect_pattern_inefficiencies(
        self, prices: pd.DataFrame, volume: pd.Series
    ) -> List[InefficencySignal]:
        """Detect technical pattern-based inefficiencies"""
        patterns = self.pattern_detector.detect_patterns(prices)
        signals = []

        for pattern in patterns:
            if pattern.type in self.config.get("enabled_patterns", []):
                signals.append(
                    InefficencySignal(
                        type=f"pattern_{pattern.type}",
                        confidence=pattern.confidence,
                        magnitude=pattern.expected_move,
                        horizon=pattern.timeframe,
                        metadata={"pattern_data": pattern.metadata},
                    )
                )

        return signals

    def _detect_behavioral_inefficiencies(
        self, prices: pd.DataFrame, volume: pd.Series, sentiment: Optional[pd.Series]
    ) -> List[InefficencySignal]:
        """Detect behavioral market inefficiencies"""
        signals = []
        returns = prices["close"].pct_change()
        vol_ratio = volume / volume.rolling(20).mean()

        # Detect overreaction to news
        if sentiment is not None:
            sentiment_change = sentiment.diff()
            price_change = returns.rolling(5).sum()

            # Price-sentiment divergence
            divergence = sentiment_change.rolling(5).mean() * price_change < 0
            if divergence.any():
                signals.append(
                    InefficencySignal(
                        type="sentiment_divergence",
                        confidence=0.7,
                        magnitude=abs(price_change[-1]),
                        horizon="medium",
                        metadata={"sentiment_change": sentiment_change[-1]},
                    )
                )

        # Other behavioral signals...
        return signals

    def _detect_liquidity_inefficiencies(
        self, prices: pd.DataFrame, volume: pd.Series
    ) -> List[InefficencySignal]:
        """Detect liquidity-based market inefficiencies"""
        # Implementation for liquidity inefficiencies
        pass
