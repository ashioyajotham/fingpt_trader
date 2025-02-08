from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """Multi-timeframe trend following with market regime awareness"""
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        
        # Timeframes for analysis
        self.timeframes = {
            'fast': 5,    # 5-minute
            'medium': 15, # 15-minute
            'slow': 60   # 1-hour
        }
        
        # Technical parameters
        self.ema_periods = {
            'short': 9,
            'medium': 21,
            'long': 55
        }
        
        # Initialize state
        self.indicators = {}
        self.signals = {}
        
    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        signals = []
        
        for symbol in self.active_pairs:
            if symbol not in market_data.get('candles', {}):
                continue
                
            # Process each timeframe
            timeframe_signals = []
            for tf_name, minutes in self.timeframes.items():
                df = self._prepare_dataframe(market_data['candles'][symbol])
                indicators = self._calculate_indicators(df, tf_name)
                
                # Store indicators
                if symbol not in self.indicators:
                    self.indicators[symbol] = {}
                self.indicators[symbol][tf_name] = indicators
                
                # Generate timeframe signal
                tf_signal = self._analyze_timeframe(indicators)
                timeframe_signals.append(tf_signal)
            
            # Combine signals across timeframes
            if timeframe_signals:
                final_signal = self._combine_timeframe_signals(timeframe_signals)
                if final_signal['strength'] > 0:
                    signals.append({
                        'symbol': symbol,
                        'direction': final_signal['direction'],
                        'strength': final_signal['strength'],
                        'timeframes': final_signal['timeframes']
                    })
                    
        return signals
        
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate technical indicators for given timeframe"""
        close = df['close']
        
        # Calculate EMAs
        emas = {}
        for name, period in self.ema_periods.items():
            emas[name] = close.ewm(span=period, adjust=False).mean()
        
        # Calculate momentum
        momentum = close.pct_change(periods=self.ema_periods['short'])
        
        # Calculate volatility
        returns = np.log(close / close.shift(1))
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        return {
            'emas': emas,
            'momentum': momentum,
            'volatility': volatility,
            'close': close
        }
