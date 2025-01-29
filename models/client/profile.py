from dataclasses import dataclass
from typing import Dict

@dataclass
class MockClientProfile:
    risk_score: int  # 1-10 scale
    investment_horizon: int  # Days
    constraints: Dict  # Trading constraints
    tax_rate: float  # Tax rate percentage
    esg_preferences: Dict  # Environmental, Social, Governance preferences
    
    def __post_init__(self):
        # Validate risk score
        if not 1 <= self.risk_score <= 10:
            raise ValueError("Risk score must be between 1 and 10")
            
        # Validate investment horizon
        if self.investment_horizon <= 0:
            raise ValueError("Investment horizon must be positive")
            
        # Set default constraints if empty
        if not self.constraints:
            self.constraints = {
                'max_position_size': 0.1,  # 10% of portfolio
                'min_position_size': 0.01,  # 1% of portfolio
                'max_leverage': 1.0  # No leverage by default
            }
            
        # Set default ESG preferences if empty
        if not self.esg_preferences:
            self.esg_preferences = {
                'environmental_score': 5,
                'social_score': 5,
                'governance_score': 5
            }