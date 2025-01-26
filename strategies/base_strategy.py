from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from models.portfolio.risk import MarketRegime, MarketRegimeDetector, CircuitBreaker

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.positions: Dict[str, float] = {}
        self.regime_detector = MarketRegimeDetector()
        self.circuit_breaker = CircuitBreaker(config.get('risk', {}).get('thresholds', {}))
        
        # Risk-based position sizing
        self.position_limits = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOL: 0.5,
            MarketRegime.STRESS: 0.25,
            MarketRegime.CRISIS: 0.0,
            MarketRegime.LOW_LIQUIDITY: 0.3
        }

    @abstractmethod
    async def process_market_data(self, data: Dict) -> None:
        """Process incoming market data"""
        pass

    @abstractmethod
    async def on_trade(self, trade: Dict) -> None:
        """Handle trade execution updates"""
        pass

    async def start(self) -> None:
        """Initialize and start strategy"""
        self.active = True
        self.last_update = datetime.now()
        await self._init_state()

    async def stop(self) -> None:
        """Stop strategy"""
        self.active = False
        await self._cleanup()

    async def update_position(self, symbol: str, quantity: float) -> None:
        """Update strategy position"""
        self.positions[symbol] = quantity

    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions.copy()

    async def _init_state(self) -> None:
        """Initialize strategy state"""
        self.positions = {}
        self.signals = []

    async def _cleanup(self) -> None:
        """Cleanup strategy resources"""
        self.positions = {}
        self.signals = []

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        required = {"symbol", "direction", "strength"}
        return all(k in signal for k in required)

    async def generate_signals(self, market_data: Dict) -> List[Dict]:
        """Generate trading signals with risk checks"""
        # Check circuit breaker first
        if self.circuit_breaker.check_conditions(market_data):
            return []  # No signals if circuit breaker triggered
            
        # Detect market regime
        current_regime = self.regime_detector.detect_regime(market_data)
        position_scale = self.position_limits[current_regime]
        
        # Generate base signals
        signals = await self._generate_base_signals(market_data)
        
        # Adjust signal sizes based on regime
        for signal in signals:
            signal['size'] *= position_scale
            signal['regime'] = current_regime.value
            
        return signals
        
    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        """To be implemented by concrete strategies"""
        raise NotImplementedError
