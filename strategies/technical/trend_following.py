
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy with multiple timeframe analysis.
    Combines technical indicators with market regime awareness.
    """
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        self.lookback_periods = {
            'short': 20,   # 20 periods for short-term trend
            'medium': 50,  # 50 periods for medium-term trend
            'long': 200    # 200 periods for long-term trend
        }
        
    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        signals = []
        for pair in self.active_pairs:
            if pair not in market_data.get('candles', {}):
                continue
                
            candles = market_data['candles'][pair]
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate technical indicators
            signals.append({
                'symbol': pair,
                'direction': self._calculate_trend_direction(df),
                'strength': self._calculate_signal_strength(df),
                'timestamp': df.iloc[-1]['timestamp']
            })
            
        return signals
        
    def _calculate_trend_direction(self, df: pd.DataFrame) -> int:
        """Calculate trend direction using multiple timeframes"""
        close = df['close'].astype(float)
        
        # Calculate EMAs for different timeframes
        ema_short = close.ewm(span=self.lookback_periods['short']).mean()
        ema_medium = close.ewm(span=self.lookback_periods['medium']).mean()
        ema_long = close.ewm(span=self.lookback_periods['long']).mean()
        
        # Determine trend direction
        short_trend = ema_short.iloc[-1] > ema_medium.iloc[-1]
        medium_trend = ema_medium.iloc[-1] > ema_long.iloc[-1]
        
        return 1 if (short_trend and medium_trend) else -1 if (not short_trend and not medium_trend) else 0

    def _calculate_signal_strength(self, df: pd.DataFrame) -> float:
        """Calculate signal strength based on trend alignment and volatility"""
        close = df['close'].astype(float)
        returns = np.log(close / close.shift(1))
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend momentum
        momentum = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        
        # Normalize strength between 0 and 1
        strength = min(abs(momentum) / (volatility * 100), 1.0)
        
        return strength
