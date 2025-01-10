from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class WeightOptimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 1.0)
        self.target_vol = self.config.get('target_volatility', 0.15)
        self.lookback_window = self.config.get('lookback_window', 252)
        
    def optimize_weights(self, 
                        returns: pd.DataFrame, 
                        method: str = 'risk_parity') -> Dict[str, float]:
        """Optimize strategy weights"""
        if method == 'risk_parity':
            return self._risk_parity(returns)
        elif method == 'min_variance':
            return self._minimum_variance(returns)
        else:
            return self._mean_variance(returns)
            
    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights"""
        cov = returns.cov()
        n = len(returns.columns)
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            risk_contrib = weights * (cov @ weights) / portfolio_vol
            return np.sum((risk_contrib - portfolio_vol/n)**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'inequality', 'fun': lambda x: x - self.min_weight},
            {'type': 'inequality', 'fun': lambda x: self.max_weight - x}
        ]
        
        result = minimize(
            risk_budget_objective,
            x0=np.ones(n)/n,
            constraints=constraints,
            method='SLSQP'
        )
        
        return dict(zip(returns.columns, result.x))
        
    def _minimum_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate minimum variance weights"""
        cov = returns.cov()
        n = len(returns.columns)
        
        def portfolio_vol(weights):
            return np.sqrt(weights.T @ cov @ weights)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'inequality', 'fun': lambda x: x - self.min_weight},
            {'type': 'inequality', 'fun': lambda x: self.max_weight - x}
        ]
        
        result = minimize(
            portfolio_vol,
            x0=np.ones(n)/n,
            constraints=constraints,
            method='SLSQP'
        )
        
        return dict(zip(returns.columns, result.x))
        
    def _mean_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean-variance optimal weights"""
        mu = returns.mean()
        cov = returns.cov()
        n = len(returns.columns)
        
        def sharpe_ratio(weights):
            port_return = np.sum(mu * weights)
            port_vol = np.sqrt(weights.T @ cov @ weights)
            return -(port_return / port_vol)
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'inequality', 'fun': lambda x: x - self.min_weight},
            {'type': 'inequality', 'fun': lambda x: self.max_weight - x}
        ]
        
        result = minimize(
            sharpe_ratio,
            x0=np.ones(n)/n,
            constraints=constraints,
            method='SLSQP'
        )
        
        return dict(zip(returns.columns, result.x))