import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from .patterns import PatternDetector

import logging
logger = logging.getLogger(__name__)


@dataclass
class InefficencySignal:
    type: str  # Type of inefficiency
    confidence: float  # Signal confidence [0-1]
    magnitude: float  # Expected price impact
    horizon: str  # Expected time horizon
    metadata: Dict  # Additional signal info


class MarketInefficencyDetector:
    """Market inefficiency detection using multiple indicators."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pattern_detector = PatternDetector(config.get("pattern_config", {}))
        self.lookback_periods = config.get(
            "lookback_periods", {"short": 5, "medium": 20, "long": 60}
        )
        self.running = False
        
    async def initialize(self):
        """Initialize detector resources"""
        try:
            self.running = True
            # Initialize pattern detector
            if hasattr(self.pattern_detector, 'initialize'):
                await self.pattern_detector.initialize()
            logger.info("Market inefficiency detector initialized")
        except Exception as e:
            logger.error(f"Detector initialization failed: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup detector resources"""
        try:
            self.running = False
            # Cleanup pattern detector
            if hasattr(self.pattern_detector, 'cleanup'):
                await self.pattern_detector.cleanup()
            logger.info("Market inefficiency detector cleaned up")
        except Exception as e:
            logger.error(f"Detector cleanup failed: {str(e)}")

    def detect_inefficiencies(self, prices, volume, sentiment=None):
        """Detect market inefficiencies with proper signal structure"""
        direction = 0
        confidence = 0.5  # Default confidence
        
        # Your existing detection code...
        
        # Add this before returning:
        magnitude = min(0.75, abs(confidence * 2))  # Scale magnitude to be between 0-1
        
        return {
            'direction': direction,
            'confidence': confidence,
            'magnitude': magnitude,  # Add this line
            'timestamp': datetime.now().isoformat()
        }

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

    def _detect_liquidity_signals(self, prices: pd.DataFrame, 
                                volume: pd.Series) -> List[Dict]:
        """Detect liquidity-based inefficiencies."""
        try:
            # Calculate volume profile
            vol_ma = volume.rolling(window=20).mean()
            vol_ratio = volume / vol_ma
            
            # Look for volume spikes
            signals = []
            for i in range(len(vol_ratio)):
                if vol_ratio.iloc[i] > 2.0:  # Volume spike threshold
                    signals.append({
                        'confidence': min(vol_ratio.iloc[i] / 4.0, 1.0),
                        'direction': 1 if prices['close'].iloc[i] > prices['open'].iloc[i] else -1,
                        'magnitude': vol_ratio.iloc[i] / 2.0,
                        'metadata': {'source': 'liquidity'}
                    })
            return signals
            
        except Exception as e:
            logger.error(f"Error in liquidity detection: {e}")
            return []  # Return empty list instead of None

    def _detect_technical_signals(self, prices: pd.DataFrame) -> List[Dict]:
        """
        Detect technical analysis based signals.
        
        Args:
            prices: DataFrame with OHLCV data
            
        Returns:
            List[Dict]: List of technical signals with:
                - confidence: Signal strength [0-1]
                - direction: 1 for long, -1 for short
                - magnitude: Expected price movement
                - metadata: Signal details
        """
        try:
            signals = []
            
            # Get price series
            close = prices['close']
            
            # 1. Moving Average Crossovers
            short_ma = close.rolling(window=5).mean()
            long_ma = close.rolling(window=20).mean()
            
            # Generate crossover signals
            if len(close) > 20:  # Ensure enough data
                # Bullish crossover
                if short_ma.iloc[-2] < long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
                    signals.append({
                        'confidence': 0.6,
                        'direction': 1,  # Long
                        'magnitude': 0.02,  # 2% expected move
                        'metadata': {'source': 'ma_crossover_bullish'}
                    })
                    
                # Bearish crossover    
                elif short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
                    signals.append({
                        'confidence': 0.6,
                        'direction': -1,  # Short
                        'magnitude': 0.02,
                        'metadata': {'source': 'ma_crossover_bearish'}
                    })
            
            # 2. RSI Signals
            rsi_period = 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Generate RSI signals
            if len(rsi) > rsi_period:
                current_rsi = rsi.iloc[-1]
                
                # Oversold signal
                if current_rsi < 30:
                    signals.append({
                        'confidence': 0.7,
                        'direction': 1,  # Long
                        'magnitude': 0.03,
                        'metadata': {'source': 'rsi_oversold', 'value': current_rsi}
                    })
                    
                # Overbought signal
                elif current_rsi > 70:
                    signals.append({
                        'confidence': 0.7,
                        'direction': -1,  # Short
                        'magnitude': 0.03,
                        'metadata': {'source': 'rsi_overbought', 'value': current_rsi}
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in technical signal detection: {e}")
            return []  # Return empty list on error
