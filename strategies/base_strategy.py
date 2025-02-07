from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from models.portfolio.risk import MarketRegime, MarketRegimeDetector, CircuitBreaker

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        """
        Initialize strategy with config and profile
        
        Args:
            config: Strategy configuration parameters
            profile: Trading profile parameters
        """
        self.config = config or {}
        self.profile = profile or {}
        self.enabled = self.config.get('enabled', True)
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
    async def process_market_data(self, market_data: Dict) -> Dict:
        """Process market data and return results"""
        pass

    @abstractmethod
    async def on_trade(self, trade_data: Dict) -> None:
        """Handle trade execution updates"""
        pass

    @abstractmethod
    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        """Generate base trading signals - must be implemented by subclasses"""
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
            if self._validate_signal(signal):
                signal['size'] *= position_scale
                signal['regime'] = current_regime.value
            
        return signals

    @abstractmethod
    async def analyze(self, pair: str, current_price: float = None, price: float = None,
                     position: float = 0, position_size: float = None,
                     holding_period: int = None, unrealized_pnl: float = None) -> Optional[str]:
        """
        Analyze market data and generate trading signals
        
        Args:
            pair: Trading pair symbol
            current_price: Current market price
            price: Alternative price input
            position: Current position size (legacy)
            position_size: Current position size (new format)
            holding_period: Days position has been held
            unrealized_pnl: Current unrealized profit/loss
            
        Returns:
            Optional[str]: Trading signal ('BUY', 'SELL', or None)
        """
        pass
        
    @abstractmethod
    async def validate(self, **kwargs) -> bool:
        """
        Validate strategy parameters and configuration
        
        Returns:
            bool: True if valid, False otherwise
        """
        pass
