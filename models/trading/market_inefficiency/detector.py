import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class MarketPattern:
    name: str
    confidence: float
    magnitude: float
    duration: int  # in periods
    description: str

class MarketInefficiencyDetector:
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.min_confidence = 0.7
        self.volatility_window = 20
        
    def detect_inefficiencies(self, 
                            prices: pd.Series, 
                            volumes: pd.Series,
                            sentiment_scores: Optional[pd.Series] = None) -> List[MarketPattern]:
        """Detect various market inefficiencies"""
        patterns = []
        
        # Microstructure patterns
        patterns.extend(self._detect_microstructure_patterns(prices, volumes))
        
        # Behavioral patterns
        patterns.extend(self._detect_behavioral_patterns(prices, volumes))
        
        # Event-driven patterns
        if sentiment_scores is not None:
            patterns.extend(self._detect_event_patterns(prices, sentiment_scores))
        
        return [p for p in patterns if p.confidence >= self.min_confidence]
    
    def _detect_microstructure_patterns(self, 
                                      prices: pd.Series,
                                      volumes: pd.Series) -> List[MarketPattern]:
        """Detect market microstructure inefficiencies"""
        patterns = []
        
        # Volume-price divergence
        returns = prices.pct_change()
        volume_ma = volumes.rolling(window=5).mean()
        price_volume_corr = returns.rolling(window=20).corr(volumes)
        
        if price_volume_corr.iloc[-1] < -0.7:
            patterns.append(MarketPattern(
                name="volume_price_divergence",
                confidence=abs(price_volume_corr.iloc[-1]),
                magnitude=returns.iloc[-1],
                duration=5,
                description="Unusual volume-price relationship detected"
            ))
            
        # Bid-ask bounce detection
        tick_changes = np.diff(prices.iloc[-100:])
        sign_changes = np.sum(np.diff(np.sign(tick_changes)) != 0)
        if sign_changes / len(tick_changes) > 0.7:
            patterns.append(MarketPattern(
                name="bid_ask_bounce",
                confidence=0.8,
                magnitude=np.std(tick_changes),
                duration=1,
                description="High frequency bid-ask bounce pattern"
            ))
            
        return patterns
    
    def _detect_behavioral_patterns(self,
                                  prices: pd.Series,
                                  volumes: pd.Series) -> List[MarketPattern]:
        """Detect behavioral market inefficiencies"""
        patterns = []
        returns = prices.pct_change()
        
        # Overreaction pattern
        volatility = returns.rolling(window=self.volatility_window).std()
        recent_vol = volatility.iloc[-1]
        avg_vol = volatility.mean()
        
        if recent_vol > 2 * avg_vol:
            patterns.append(MarketPattern(
                name="overreaction",
                confidence=min(1.0, recent_vol / avg_vol - 1),
                magnitude=returns.iloc[-1],
                duration=self.volatility_window,
                description="Market overreaction detected"
            ))
            
        # Mean reversion potential
        z_score = (prices - prices.rolling(window=20).mean()) / prices.rolling(window=20).std()
        if abs(z_score.iloc[-1]) > 2:
            patterns.append(MarketPattern(
                name="mean_reversion",
                confidence=min(1.0, abs(z_score.iloc[-1]) / 3),
                magnitude=z_score.iloc[-1],
                duration=20,
                description="Potential mean reversion opportunity"
            ))
            
        return patterns
        
    def _detect_event_patterns(self,
                             prices: pd.Series,
                             sentiment_scores: pd.Series) -> List[MarketPattern]:
        """Detect event-driven inefficiencies"""
        patterns = []
        returns = prices.pct_change()
        
        # Sentiment-price divergence
        sentiment_ma = sentiment_scores.rolling(window=5).mean()
        price_ma = returns.rolling(window=5).mean()
        
        if sentiment_ma.iloc[-1] > 0.7 and price_ma.iloc[-1] < 0:
            patterns.append(MarketPattern(
                name="sentiment_price_divergence",
                confidence=sentiment_ma.iloc[-1],
                magnitude=abs(price_ma.iloc[-1]),
                duration=5,
                description="Positive sentiment with negative price movement"
            ))
        elif sentiment_ma.iloc[-1] < 0.3 and price_ma.iloc[-1] > 0:
            patterns.append(MarketPattern(
                name="sentiment_price_divergence",
                confidence=1 - sentiment_ma.iloc[-1],
                magnitude=abs(price_ma.iloc[-1]),
                duration=5,
                description="Negative sentiment with positive price movement"
            ))
            
        return patterns
    
    def get_composite_score(self, patterns: List[MarketPattern]) -> float:
        """Calculate composite inefficiency score"""
        if not patterns:
            return 0.0
            
        scores = [p.confidence * p.magnitude for p in patterns]
        weights = [p.confidence for p in patterns]
        
        return np.average(scores, weights=weights)
