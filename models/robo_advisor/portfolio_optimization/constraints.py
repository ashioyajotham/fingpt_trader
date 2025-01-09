from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import LinearConstraint

@dataclass
class PortfolioConstraints:
    min_weight: float = 0.0
    max_weight: float = 0.4
    min_assets: int = 5
    max_sector_weight: float = 0.4
    turnover_limit: float = 0.2
    risk_budget: Optional[Dict[str, float]] = None

class ConstraintGenerator:
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        
    def generate_weight_constraints(self, n_assets: int) -> List[LinearConstraint]:
        """Generate basic portfolio weight constraints"""
        # Sum of weights = 1
        eq_constraint = LinearConstraint(
            np.ones(n_assets),
            lb=1.0,
            ub=1.0
        )
        
        # Individual weight bounds
        bounds = LinearConstraint(
            np.eye(n_assets),
            lb=[self.constraints.min_weight] * n_assets,
            ub=[self.constraints.max_weight] * n_assets
        )
        
        return [eq_constraint, bounds]
        
    def generate_sector_constraints(self, 
                                  sector_mappings: Dict[str, str]) -> List[LinearConstraint]:
        """Generate sector-based constraints"""
        unique_sectors = list(set(sector_mappings.values()))
        n_assets = len(sector_mappings)
        
        sector_constraints = []
        for sector in unique_sectors:
            # Create sector mask
            sector_mask = np.zeros(n_assets)
            for i, (asset, asset_sector) in enumerate(sector_mappings.items()):
                if asset_sector == sector:
                    sector_mask[i] = 1
                    
            sector_constraints.append(
                LinearConstraint(
                    sector_mask,
                    lb=0.0,
                    ub=self.constraints.max_sector_weight
                )
            )
            
        return sector_constraints
        
    def generate_turnover_constraints(self, 
                                    current_weights: np.ndarray) -> LinearConstraint:
        """Generate turnover constraint"""
        n_assets = len(current_weights)
        
        return LinearConstraint(
            np.abs(np.eye(n_assets) - current_weights.reshape(-1, 1)),
            lb=0.0,
            ub=self.constraints.turnover_limit
        )
        
    def generate_risk_budget_constraints(self, 
                                       covariance: np.ndarray) -> List[LinearConstraint]:
        """Generate risk budgeting constraints"""
        if not self.constraints.risk_budget:
            return []
            
        n_assets = covariance.shape[0]
        risk_budget = np.array(list(self.constraints.risk_budget.values()))
        
        def risk_budget_constraint(weights):
            portfolio_risk = np.sqrt(weights.T @ covariance @ weights)
            marginal_risk = (covariance @ weights) / portfolio_risk
            risk_contributions = weights * marginal_risk
            return risk_contributions - risk_budget * portfolio_risk
            
        return [{'type': 'eq', 'fun': risk_budget_constraint}]
