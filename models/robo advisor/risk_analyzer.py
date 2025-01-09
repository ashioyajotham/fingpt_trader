import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    volatility: float
    var_95: float
    max_drawdown: float
    sharpe_ratio: float
    concentration_risk: float

class RiskAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.max_concentration = 0.4
        self.volatility_threshold = 0.25
        
    def calculate_portfolio_risk(self,
                               returns: pd.DataFrame,
                               weights: Dict[str, float]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        weight_array = np.array(list(weights.values()))
        portfolio_returns = returns.dot(weight_array)
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        sharpe = self._calculate_sharpe_ratio(portfolio_returns)
        concentration = self._calculate_concentration_risk(weights)
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            concentration_risk=concentration
        )
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration risk"""
        return max(weights.values()) / self.max_concentration
        
    def generate_risk_alerts(self, metrics: RiskMetrics) -> List[Dict]:
        """Generate risk alerts based on metrics"""
        alerts = []
        
        if metrics.volatility > self.volatility_threshold:
            alerts.append({
                'level': 'high',
                'type': 'volatility',
                'message': f'Portfolio volatility ({metrics.volatility:.2%}) exceeds threshold'
            })
            
        if metrics.concentration_risk > 1:
            alerts.append({
                'level': 'medium',
                'type': 'concentration',
                'message': 'Portfolio shows high concentration risk'
            })
            
        if metrics.sharpe_ratio < 0.5:
            alerts.append({
                'level': 'warning',
                'type': 'performance',
                'message': f'Low Sharpe ratio ({metrics.sharpe_ratio:.2f})'
            })
            
        return alerts