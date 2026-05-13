from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_positions: int = 5
    max_positions: int = 20
    target_volatility: Optional[float] = None
    sector_constraints: Optional[Dict[str, float]] = None


class PortfolioOptimizer:
    def __init__(self, config: Dict):
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        self.default_constraints = OptimizationConstraints(
            **config.get("constraints", {})
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> Dict:
        """Optimize portfolio weights for maximum Sharpe ratio"""
        constraints = constraints or self.default_constraints
        if isinstance(returns, np.ndarray):
            returns = pd.DataFrame(
                returns,
                columns=[f"asset_{i}" for i in range(returns.shape[1])],
            )

        # Calculate expected returns and covariance
        mu = returns.mean() * 252  # Annualized returns
        sigma = returns.cov() * 252  # Annualized covariance

        # Setup optimization
        n_assets = len(returns.columns)
        x0 = np.ones(n_assets) / n_assets  # Equal weight start

        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            port_return = np.sum(mu * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol <= 0:
                return 0.0
            sharpe = (port_return - self.risk_free_rate) / port_vol
            return -sharpe

        # Optimization constraints
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            {
                "type": "ineq",
                "fun": lambda x: x - constraints.min_weight,
            },  # Min weights
            {
                "type": "ineq",
                "fun": lambda x: constraints.max_weight - x,
            },  # Max weights
        ]

        # Add position limits if specified
        if constraints.min_positions:
            constraints_list.append(
                {
                    "type": "ineq",
                    "fun": lambda x: np.sum(x > 0.0001) - constraints.min_positions,
                }
            )

        # Add volatility target if specified
        if constraints.target_volatility:
            constraints_list.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sqrt(np.dot(x.T, np.dot(sigma, x)))
                    - constraints.target_volatility,
                }
            )

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints_list,
            bounds=[
                (constraints.min_weight, constraints.max_weight)
                for _ in range(n_assets)
            ],
        )

        if result.success:
            weights = result.x
        else:
            weights = np.ones(n_assets) / n_assets
            weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
            weights = weights / weights.sum()

        # Calculate portfolio metrics
        port_return = np.sum(mu * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return {
            "weights": dict(zip(returns.columns, weights)),
            "metrics": {
                "expected_return": port_return,
                "volatility": port_vol,
                "sharpe_ratio": sharpe,
            },
            "diagnostics": {"success": result.success, "message": result.message},
        }
