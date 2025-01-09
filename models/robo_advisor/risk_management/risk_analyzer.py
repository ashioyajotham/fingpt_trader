import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class RiskMetrics:
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: Optional[float] = None
    correlation_matrix: Optional[np.ndarray] = None

class RiskAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.annual_factor = np.sqrt(252)  # For daily data
        
    def calculate_portfolio_risk(self, 
                               returns: pd.DataFrame,
                               weights: np.ndarray,
                               benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        portfolio_returns = returns.dot(weights)
        
        # Basic risk metrics
        volatility = self._calculate_volatility(portfolio_returns)
        var_95 = self._calculate_var(portfolio_returns)
        cvar_95 = self._calculate_cvar(portfolio_returns)
        max_dd = self._calculate_max_drawdown(portfolio_returns)
        
        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio(portfolio_returns)
        sortino = self._calculate_sortino_ratio(portfolio_returns)
        
        # Market-relative metrics
        beta = None
        if benchmark_returns is not None:
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            
        # Portfolio correlations
        corr_matrix = returns.corr().values
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta=beta,
            correlation_matrix=corr_matrix
        )
        
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * self.annual_factor
        
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return -np.percentile(returns, (1 - confidence) * 100) * self.annual_factor
        
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return -returns[returns < -var/self.annual_factor].mean() * self.annual_factor
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return drawdowns.min()
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * self.annual_factor
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        return np.mean(excess_returns) / np.std(downside_returns) * self.annual_factor
        
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        covar = np.cov(returns, benchmark_returns)[0,1]
        benchmark_var = np.var(benchmark_returns)
        return covar / benchmark_var if benchmark_var != 0 else 1.0
