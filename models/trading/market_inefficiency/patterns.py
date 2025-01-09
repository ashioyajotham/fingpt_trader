import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    MICROSTRUCTURE = "microstructure"
    EVENT_DRIVEN = "event_driven"

@dataclass
class PatternConfig:
    lookback_window: int
    threshold: float
    min_confidence: float
    decay_factor: float = 0.95

class TradingPattern:
    def __init__(self, 
                 name: str, 
                 pattern_type: PatternType,
                 config: PatternConfig):
        self.name = name
        self.type = pattern_type
        self.config = config
        
    def detect(self, 
              prices: pd.Series, 
              volumes: pd.Series, 
              additional_data: Optional[Dict] = None) -> Tuple[bool, float, str]:
        """Base pattern detection method"""
        raise NotImplementedError

class PriceActionPattern(TradingPattern):
    def detect(self, 
              prices: pd.Series, 
              volumes: pd.Series, 
              additional_data: Optional[Dict] = None) -> Tuple[bool, float, str]:
        """Detect price action based patterns"""
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.config.lookback_window).std()
        
        # Pattern specific logic
        if self.name == "momentum_reversal":
            return self._detect_momentum_reversal(returns, volatility)
        elif self.name == "volatility_breakout":
            return self._detect_volatility_breakout(returns, volatility)
            
        return False, 0.0, ""
        
    def _detect_momentum_reversal(self, 
                                returns: pd.Series, 
                                volatility: pd.Series) -> Tuple[bool, float, str]:
        """Detect momentum reversal patterns"""
        momentum = returns.rolling(window=10).sum()
        recent_momentum = momentum.iloc[-1]
        
        if abs(recent_momentum) > 2 * volatility.iloc[-1]:
            confidence = min(1.0, abs(recent_momentum) / (3 * volatility.iloc[-1]))
            direction = "upward" if recent_momentum < 0 else "downward"
            return True, confidence, f"Potential {direction} reversal detected"
            
        return False, 0.0, ""
        
    def _detect_volatility_breakout(self, 
                                  returns: pd.Series, 
                                  volatility: pd.Series) -> Tuple[bool, float, str]:
        """Detect volatility breakout patterns"""
        vol_ratio = volatility.iloc[-1] / volatility.iloc[-20:].mean()
        
        if vol_ratio > self.config.threshold:
            confidence = min(1.0, (vol_ratio - self.config.threshold) / 2)
            return True, confidence, "Volatility breakout detected"
            
        return False, 0.0, ""

class BehavioralPattern(TradingPattern):
    def detect(self, 
              prices: pd.Series, 
              volumes: pd.Series, 
              additional_data: Optional[Dict] = None) -> Tuple[bool, float, str]:
        """Detect behavioral patterns"""
        if self.name == "panic_selling":
            return self._detect_panic_selling(prices, volumes)
        elif self.name == "fomo_buying":
            return self._detect_fomo_buying(prices, volumes)
            
        return False, 0.0, ""
        
    def _detect_panic_selling(self, 
                            prices: pd.Series, 
                            volumes: pd.Series) -> Tuple[bool, float, str]:
        """Detect panic selling patterns"""
        returns = prices.pct_change()
        volume_ratio = volumes.iloc[-1] / volumes.rolling(window=20).mean().iloc[-1]
        
        if returns.iloc[-1] < -2 * returns.std() and volume_ratio > 2:
            confidence = min(1.0, volume_ratio / 4)
            return True, confidence, "Panic selling detected"
            
        return False, 0.0, ""
        
    def _detect_fomo_buying(self, 
                          prices: pd.Series, 
                          volumes: pd.Series) -> Tuple[bool, float, str]:
        """Detect FOMO buying patterns"""
        returns = prices.pct_change()
        volume_ma = volumes.rolling(window=5).mean()
        
        if returns.tail(3).mean() > 0 and volume_ma.iloc[-1] > 2 * volume_ma.iloc[-20:].mean():
            confidence = min(1.0, returns.tail(3).mean() / returns.std())
            return True, confidence, "FOMO buying detected"
            
        return False, 0.0, ""

class MarketMicrostructurePattern(TradingPattern):
    def detect(self, 
              prices: pd.Series, 
              volumes: pd.Series, 
              additional_data: Optional[Dict] = None) -> Tuple[bool, float, str]:
        """Detect market microstructure patterns"""
        if self.name == "order_imbalance":
            return self._detect_order_imbalance(volumes)
        elif self.name == "liquidity_gap":
            return self._detect_liquidity_gap(prices, volumes)
            
        return False, 0.0, ""
        
    def _detect_order_imbalance(self, volumes: pd.Series) -> Tuple[bool, float, str]:
        """Detect order flow imbalances"""
        vol_std = volumes.rolling(window=20).std()
        vol_mean = volumes.rolling(window=20).mean()
        zscore = (volumes - vol_mean) / vol_std
        
        if abs(zscore.iloc[-1]) > self.config.threshold:
            confidence = min(1.0, abs(zscore.iloc[-1]) / (self.config.threshold * 2))
            return True, confidence, "Significant order imbalance detected"
            
        return False, 0.0, ""
        
    def _detect_liquidity_gap(self, 
                            prices: pd.Series, 
                            volumes: pd.Series) -> Tuple[bool, float, str]:
        """Detect liquidity gaps"""
        volume_ratio = volumes.iloc[-1] / volumes.rolling(window=20).mean().iloc[-1]
        price_gap = abs(prices.diff().iloc[-1])
        
        if volume_ratio < 0.5 and price_gap > prices.std() * 2:
            confidence = min(1.0, (1/volume_ratio - 1) / 2)
            return True, confidence, "Liquidity gap detected"
            
        return False, 0.0, ""

def create_pattern(name: str, 
                  pattern_type: PatternType, 
                  config: PatternConfig) -> TradingPattern:
    """Factory function to create pattern instances"""
    pattern_classes = {
        PatternType.TECHNICAL: PriceActionPattern,
        PatternType.BEHAVIORAL: BehavioralPattern,
        PatternType.MICROSTRUCTURE: MarketMicrostructurePattern
    }
    
    pattern_class = pattern_classes.get(pattern_type)
    if pattern_class:
        return pattern_class(name, pattern_type, config)
    raise ValueError(f"Unsupported pattern type: {pattern_type}")
