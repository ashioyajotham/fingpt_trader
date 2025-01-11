from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint
from ..base_strategy import BaseStrategy

class InefficiencyStrategy(BaseStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.window = self.config.get('window', 20)
        self.z_threshold = self.config.get('z_threshold', 2.0)
        # Convert list pairs to tuples for hashability
        self.pairs = [tuple(pair) for pair in self.config.get('pairs', [])]
        self.position_data = {}
        self.cointegration_scores = {}
        
    async def process_market_data(self, data: Dict) -> None:
        """Process market data updates"""
        if 'close' in data:
            symbol = data['symbol']
            self.market_data[symbol] = data
            await self._update_cointegration()
            
    async def on_trade(self, trade: Dict) -> None:
        """Handle trade updates"""
        symbol = trade.get('symbol')
        if symbol in self.positions:
            self.positions[symbol].update(trade)
            
    async def generate_signals(self) -> List[Dict]:
        """Generate trading signals based on inefficiencies"""
        signals = []
        for pair in self.pairs:
            if pair in self.cointegration_scores:
                signal = await self._analyze_pair(pair)
                if signal:
                    signals.append(signal)
        return signals
        
    async def _analyze_pair(self, pair: Tuple[str, str]) -> Optional[Dict]:
        """Analyze trading pair for inefficiencies"""
        symbol1, symbol2 = pair
        if not (self._has_sufficient_data(symbol1) and 
                self._has_sufficient_data(symbol2)):
            return None
            
        spread = self._calculate_spread(symbol1, symbol2)
        z_score = self._calculate_zscore(spread)
        
        if abs(z_score) > self.z_threshold:
            return {
                'pair': pair,
                'z_score': z_score,
                'direction': 'long' if z_score < 0 else 'short',
                'strength': abs(z_score) / self.z_threshold,
                'timestamp': pd.Timestamp.now()
            }
        return None
        
    def _calculate_spread(self, symbol1: str, symbol2: str) -> np.ndarray:
        """Calculate price spread between pairs"""
        prices1 = self.market_data[symbol1]['close'][-self.window:]
        prices2 = self.market_data[symbol2]['close'][-self.window:]
        return np.array(prices1) - np.array(prices2)
        
    def _calculate_zscore(self, spread: np.ndarray) -> float:
        """Calculate z-score of spread"""
        return (spread[-1] - np.mean(spread)) / np.std(spread)
        
    def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if enough market data is available"""
        return (symbol in self.market_data and 
                len(self.market_data[symbol]['close']) >= self.window)