from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MockClientProfile:
    """
    Mock client profile for testing and development.
    
    Attributes:
        risk_score (float): Client's risk tolerance (1-10)
        investment_horizon (int): Investment timeframe in years
        constraints (Dict): Trading/portfolio constraints
        tax_rate (float): Client's tax rate for harvesting
        esg_preferences (Dict): Environmental, Social, Governance preferences
    """
    risk_score: float
    investment_horizon: int
    constraints: Dict
    tax_rate: float
    esg_preferences: Dict

    def __post_init__(self):
        # Validate risk score
        if not 1 <= self.risk_score <= 10:
            raise ValueError("Risk score must be between 1 and 10")
        
        # Validate tax rate
        if not 0 <= self.tax_rate <= 1:
            raise ValueError("Tax rate must be between 0 and 1")