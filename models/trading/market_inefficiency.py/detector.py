import numpy as np
from typing import Dict, List

class MarketInefficiencyDetector:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def detect_microstructure_patterns(self, price_data: np.ndarray, 
                                     volume_data: np.ndarray) -> Dict[str, float]:
        """Detect market microstructure inefficiencies"""
        patterns = {
            'price_impact': self._calculate_price_impact(price_data, volume_data),
            'bid_ask_bounce': self._detect_bid_ask_bounce(price_data),
            'order_imbalance': self._calculate_order_imbalance(volume_data)
        }
        return patterns
    
    def detect_psychology_patterns(self, price_data: np.ndarray) -> Dict[str, float]:
        """Detect trader psychology patterns"""
        return {
            'momentum': self._calculate_momentum(price_data),
            'mean_reversion': self._calculate_mean_reversion(price_data),
            'overreaction': self._detect_overreaction(price_data)
        }
        
    def _calculate_price_impact(self, prices: np.ndarray, 
                              volumes: np.ndarray) -> float:
        """Calculate price impact ratio"""
        returns = np.diff(np.log(prices))
        volume_ratio = volumes[1:] / volumes[:-1]
        return np.corrcoef(returns, volume_ratio)[0,1]
    
    def _detect_bid_ask_bounce(self, prices: np.ndarray) -> float:
        """Measure bid-ask bounce effect"""
        return np.mean(np.abs(np.diff(prices)))
    
    def _calculate_order_imbalance(self, volumes: np.ndarray) -> float:
        """Calculate order flow imbalance"""
        return np.std(volumes) / np.mean(volumes)
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        returns = np.diff(np.log(prices))
        return np.sum(returns[-self.lookback_period:])
    
    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        ma = np.mean(prices[-self.lookback_period:])
        return (prices[-1] - ma) / ma
    
    def _detect_overreaction(self, prices: np.ndarray) -> float:
        """Detect price overreaction"""
        returns = np.diff(np.log(prices))
        return np.percentile(np.abs(returns), 95)