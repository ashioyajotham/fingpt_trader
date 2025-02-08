from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from models.portfolio.risk import MarketRegime, MarketRegimeDetector, CircuitBreaker
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Enhanced base strategy with market data and position management.
    All other strategies inherit from this."""
    
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
        
        self.market_data = {}
        self.active_pairs = config.get('pairs', ['BTCUSDT', 'ETHUSDT'])
        self.min_position = config.get('min_position_size', 0.01)
        self.max_position = config.get('max_position_size', 0.2)
        self.initial_balance = config.get('initial_balance', 10000.0)

        # Add market data handling
        self.market_state = {
            'orderbooks': {},
            'trades': {},
            'candles': {},
            'sentiment': {}
        }
        
        # Trading parameters from config
        self.risk_params = self.config.get('risk', {})
        self.trade_params = self.config.get('trading', {})
        
        # Performance tracking
        self.trades_history = []
        self.performance_metrics = {}

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

    async def on_market_data(self, data: Dict) -> None:
        """Handle incoming market data"""
        self.market_data.update(data)
        await self.analyze()

    async def generate_signal(self) -> Optional[Dict]:
        """Generate trading signal based on analysis"""
        raise NotImplementedError("Subclasses must implement generate_signal()")

    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength and account balance"""
        confidence = signal.get('confidence', 0)
        max_size = min(
            self.max_position * self.initial_balance,
            self.initial_balance * signal.get('strength', 0.1)
        )
        return max(self.min_position * self.initial_balance, max_size * confidence)

    async def process_market_data(self, market_data: Dict) -> Dict:
        """Process and store market data"""
        for data_type, data in market_data.items():
            if data_type in self.market_state:
                self.market_state[data_type].update(data)
        
        # Update market regime
        current_regime = self.regime_detector.detect_regime(market_data)
        self.market_state['regime'] = current_regime
        
        return await self._analyze_market_state()

    async def _analyze_market_state(self) -> Dict:
        """Analyze current market state"""
        analysis = {
            'regime': self.market_state['regime'],
            'volatility': self._calculate_volatility(),
            'liquidity': self._calculate_liquidity(),
            'sentiment': self._aggregate_sentiment()
        }
        return analysis

    def _calculate_volatility(self) -> float:
        """Calculate current market volatility"""
        if not self.market_state['candles']:
            return 0.0
            
        returns = []
        for pair in self.active_pairs:
            if pair in self.market_state['candles']:
                candles = self.market_state['candles'][pair]
                closes = [float(c[4]) for c in candles[-20:]]  # Last 20 closes
                if len(closes) > 1:
                    returns.append(np.std(np.diff(np.log(closes))))
                    
        return np.mean(returns) if returns else 0.0

    def _calculate_liquidity(self) -> float:
        """Calculate market liquidity score"""
        if not self.market_state['orderbooks']:
            return 0.0
            
        liquidity_scores = []
        for pair in self.active_pairs:
            if pair in self.market_state['orderbooks']:
                ob = self.market_state['orderbooks'][pair]
                spread = (float(ob['asks'][0][0]) - float(ob['bids'][0][0])) / float(ob['bids'][0][0])
                depth = sum(float(level[1]) for level in ob['bids'][:5] + ob['asks'][:5])
                liquidity_scores.append(depth / (spread + 1e-10))
                
        return np.mean(liquidity_scores) if liquidity_scores else 0.0

    def _aggregate_sentiment(self) -> float:
        """Aggregate sentiment signals"""
        if not self.market_state['sentiment']:
            return 0.0
            
        sentiments = []
        for pair in self.active_pairs:
            if pair in self.market_state['sentiment']:
                sentiments.append(self.market_state['sentiment'][pair])
                
        return np.mean(sentiments) if sentiments else 0.0
