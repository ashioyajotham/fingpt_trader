from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from services.base_service import BaseService


class RoboAdvisorService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.risk_profile = self.config.get("risk_profile", "moderate")
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)
        self.max_position_size = self.config.get("max_position_size", 0.2)

    async def _setup(self) -> None:
        """Initialize portfolio optimization"""
        self.current_portfolio = {}
        self.target_weights = {}

    async def optimize_portfolio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> Dict:
        """Modern Portfolio Theory Optimization"""
        n_assets = returns.shape[1]
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns.T)

        def objective(weights):
            port_return = np.sum(mu * weights)
            port_risk = np.sqrt(weights.T @ sigma @ weights)
            sharpe = (port_return - risk_free_rate) / port_risk
            return -sharpe  # Minimize negative Sharpe

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: self.max_position_size - w},
        ]

        bounds = [(0, self.max_position_size) for _ in range(n_assets)]
        result = minimize(
            objective,
            x0=np.array([1 / n_assets] * n_assets),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return {
            "weights": result.x,
            "expected_return": np.sum(mu * result.x),
            "volatility": np.sqrt(result.x.T @ sigma @ result.x),
            "sharpe": -result.fun,
        }
