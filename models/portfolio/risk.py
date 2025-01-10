from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

@dataclass
class RiskMetrics:
    volatility: float
    var: float  # Value at Risk
    es: float   # Expected Shortfall
    beta: float
    tracking_error: Optional[float] = None

class RiskAnalyzer:
    def __init__(self, config: Dict):
        self.confidence_level = config.get('confidence_level', 0.95)
        self.lookback_window = config.get('lookback_window', 252)
        self.use_ewma = config.get('use_ewma', True)
        
    def calculate_portfolio_risk(self,
                               returns: pd.Series,
                               weights: Dict[str, float],
                               benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Calculate portfolio returns if weights provided
        if isinstance(returns, pd.DataFrame):
            port_returns = (returns * pd.Series(weights)).sum(axis=1)
        else:
            port_returns = returns
            
        # Calculate volatility
        if self.use_ewma:
            vol = self._calculate_ewma_volatility(port_returns)
        else:
            vol = port_returns.std() * np.sqrt(252)
            
        # Calculate VaR and ES
        var = self._calculate_var(port_returns)
        es = self._calculate_expected_shortfall(port_returns)
        
        # Calculate beta if benchmark provided
        beta = (self._calculate_beta(port_returns, benchmark_returns) 
               if benchmark_returns is not None else None)
        
        # Calculate tracking error if benchmark provided
        tracking_error = (self._calculate_tracking_error(port_returns, benchmark_returns)
                         if benchmark_returns is not None else None)
        
        return RiskMetrics(
            volatility=vol,
            var=var,
            es=es,
            beta=beta if beta is not None else 0,
            tracking_error=tracking_error
        )
    
    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """Calculate EWMA volatility"""
        lambda_param = 0.94
        return np.sqrt(252) * np.sqrt(
            returns.ewm(alpha=1-lambda_param).var().iloc[-1]
        )
    
    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk"""
        return -np.percentile(returns, 100 * (1 - self.confidence_level))
    
    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self._calculate_var(returns)
        return -returns[returns <= -var].mean()
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        covar = returns.cov(benchmark_returns)
        benchmark_var = benchmark_returns.var()
        return covar / benchmark_var if benchmark_var != 0 else 0
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        return (returns - benchmark_returns).std() * np.sqrt(252)
