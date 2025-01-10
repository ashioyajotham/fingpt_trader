from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService

class PerformanceTracker(BaseService):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.trades = []
        self.portfolio_values = []
        self.benchmark_data = {}

    async def _setup(self) -> None:
        self.benchmark_symbol = self.config.get('benchmark', 'SPY')
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
    async def add_trade(self, trade: Dict) -> None:
        """Record new trade"""
        self.trades.append({
            'timestamp': datetime.now(),
            **trade
        })
        
    async def calculate_returns(self) -> Dict:
        """Calculate portfolio returns metrics"""
        df = pd.DataFrame(self.portfolio_values)
        returns = {
            'total_return': self._calculate_total_return(df),
            'sharpe_ratio': self._calculate_sharpe_ratio(df),
            'max_drawdown': self._calculate_max_drawdown(df),
            'volatility': self._calculate_volatility(df)
        }
        return returns
        
    def _calculate_total_return(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        return (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
        
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        returns = df['value'].pct_change().dropna()
        excess_returns = returns - self.risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        peak = df['value'].expanding(min_periods=1).max()
        drawdown = (df['value'] - peak) / peak
        return drawdown.min()
        
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        returns = df['value'].pct_change().dropna()
        return returns.std() * np.sqrt(252)