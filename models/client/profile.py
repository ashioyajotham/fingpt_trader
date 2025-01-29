from dataclasses import dataclass
from typing import Dict

@dataclass
class MockClientProfile:
    risk_score: float
    investment_horizon: int
    constraints: Dict
    tax_rate: float
    esg_preferences: Dict