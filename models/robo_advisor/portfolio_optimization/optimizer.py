import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from .constraints import ConstraintGenerator, PortfolioConstraints

class PortfolioOptimizer:
    def __init__(self, 
                 risk_tolerance: float = 0.5,
                 constraints: Optional[PortfolioConstraints] = None):
        self.risk_tolerance = risk_tolerance
        self.constraints = constraints or PortfolioConstraints()
        self.constraint_generator = ConstraintGenerator(self.constraints)
        
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         sentiments: Optional[Dict[str, float]] = None,
                         sector_mappings: Optional[Dict[str, str]] = None,
                         current_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Optimize portfolio weights using returns and optional sentiment"""
        
        # Calculate basic return and risk metrics
        exp_returns = returns.mean()
        covariance = returns.cov()
        
        # Adjust expected returns with sentiment if available
        if sentiments:
            exp_returns = self._adjust_returns_with_sentiment(exp_returns, sentiments)
            
        # Generate constraints
        constraints = self._generate_constraints(
            returns.shape[1],
            sector_mappings,
            current_weights
        )
        
        # Optimize portfolio
        result = self._optimize(exp_returns, covariance, constraints)
        
        # Calculate metrics for the optimal portfolio
        metrics = self._calculate_portfolio_metrics(
            result.x, exp_returns, covariance
        )
        
        return result.x, metrics
        
    def _adjust_returns_with_sentiment(self,
                                     returns: pd.Series,
                                     sentiments: Dict[str, float]) -> pd.Series:
        """Adjust expected returns based on sentiment scores"""
        adjusted_returns = returns.copy()
        
        for asset in returns.index:
            if asset in sentiments:
                sentiment_score = sentiments[asset]
                # Scale sentiment impact based on strength
                sentiment_impact = (sentiment_score - 0.5) * 0.1
                adjusted_returns[asset] *= (1 + sentiment_impact)
                
        return adjusted_returns
        
    def _generate_constraints(self,
                            n_assets: int,
                            sector_mappings: Optional[Dict[str, str]] = None,
                            current_weights: Optional[np.ndarray] = None) -> List:
        """Generate all portfolio constraints"""
        constraints = []
        
        # Basic weight constraints
        constraints.extend(
            self.constraint_generator.generate_weight_constraints(n_assets)
        )
        
        # Sector constraints if sector mappings provided
        if sector_mappings:
            constraints.extend(
                self.constraint_generator.generate_sector_constraints(sector_mappings)
            )
            
        # Turnover constraints if current weights provided
        if current_weights is not None:
            constraints.append(
                self.constraint_generator.generate_turnover_constraints(current_weights)
            )
            
        return constraints
        
    def _optimize(self,
                 exp_returns: pd.Series,
                 covariance: pd.DataFrame,
                 constraints: List) -> minimize:
        """Run portfolio optimization"""
        
        def objective(weights):
            portfolio_return = np.sum(weights * exp_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            utility = portfolio_return - self.risk_tolerance * portfolio_risk
            return -utility  # Minimize negative utility
            
        # Initial guess - equal weights
        n_assets = len(exp_returns)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(self.constraints.min_weight, self.constraints.max_weight)] * n_assets
        )
        
        return result
        
    def _calculate_portfolio_metrics(self,
                                   weights: np.ndarray,
                                   exp_returns: pd.Series,
                                   covariance: pd.DataFrame) -> Dict:
        """Calculate metrics for the optimized portfolio"""
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': portfolio_return / portfolio_risk,
            'weights': dict(zip(exp_returns.index, weights)),
            'diversification_score': 1 - np.sum(weights ** 2)  # 1 - HHI
        }
