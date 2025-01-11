from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class CombinationMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    STRENGTH_WEIGHTED = "strength_weighted"


@dataclass
class CombinedSignal:
    symbol: str
    direction: float  # -1 to 1
    strength: float  # 0 to 1
    confidence: float
    sources: List[str]
    weights: Dict[str, float]


class SignalCombiner:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.method = CombinationMethod(self.config.get("method", "weighted_average"))
        self.min_confidence = self.config.get("min_confidence", 0.3)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)

    def combine_signals(
        self, signals: List[Dict], weights: Optional[Dict[str, float]] = None
    ) -> List[CombinedSignal]:
        """Combine multiple strategy signals"""
        if not signals:
            return []

        # Group signals by symbol
        symbol_signals = self._group_by_symbol(signals)

        combined_signals = []
        for symbol, signal_group in symbol_signals.items():
            combined = self._combine_symbol_signals(signal_group, weights)
            if combined and combined.confidence >= self.min_confidence:
                combined_signals.append(combined)

        return combined_signals

    def _combine_symbol_signals(
        self, signals: List[Dict], weights: Optional[Dict[str, float]] = None
    ) -> Optional[CombinedSignal]:
        """Combine signals for single symbol"""
        if not signals:
            return None

        if self.method == CombinationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_combine(signals, weights)
        elif self.method == CombinationMethod.MAJORITY_VOTE:
            return self._majority_vote_combine(signals)
        else:
            return self._strength_weighted_combine(signals, weights)

    def _weighted_average_combine(
        self, signals: List[Dict], weights: Optional[Dict[str, float]] = None
    ) -> CombinedSignal:
        """Combine using weighted average"""
        total_weight = 0
        weighted_direction = 0
        weighted_strength = 0

        sources = []
        signal_weights = {}

        for signal in signals:
            strategy = signal["strategy"]
            weight = weights.get(strategy, 1.0) if weights else 1.0

            weighted_direction += signal["direction"] * weight
            weighted_strength += signal["strength"] * weight
            total_weight += weight

            sources.append(strategy)
            signal_weights[strategy] = weight

        return CombinedSignal(
            symbol=signals[0]["symbol"],
            direction=weighted_direction / total_weight,
            strength=weighted_strength / total_weight,
            confidence=self._calculate_confidence(signals),
            sources=sources,
            weights=signal_weights,
        )
