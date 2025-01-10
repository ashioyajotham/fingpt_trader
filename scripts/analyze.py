import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.config import ConfigManager

class PerformanceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        
    def analyze_returns(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        stats = {
            'total_return': self._calculate_total_return(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'volatility': returns.std() * np.sqrt(252)
        }
        return stats