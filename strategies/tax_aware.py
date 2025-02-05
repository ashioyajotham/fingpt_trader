from typing import Dict
from models.client.profile import MockClientProfile
from .base_strategy import BaseStrategy

class TaxAwareStrategy(BaseStrategy):
    def __init__(self, config: Dict, profile: MockClientProfile = None):
        super().__init__(config)
        self.profile = profile
        self.tax_rate = profile.tax_rate if profile else 0.25
        self.holding_periods = {}  # Track position holding periods
        
    async def generate_signals(self, market_data: Dict) -> Dict:
        """Generate trading signals considering tax implications"""
        signals = {}
        try:
            # Consider holding period and tax implications
            for symbol, data in market_data.items():
                tax_adjusted_return = self._calculate_tax_adjusted_return(
                    symbol,
                    data.get('returns', 0)
                )
                signals[symbol] = {
                    'action': 'hold',  # Default to hold
                    'score': tax_adjusted_return,
                    'metadata': {
                        'tax_impact': self._estimate_tax_impact(symbol),
                        'holding_period': self.holding_periods.get(symbol, 0)
                    }
                }
            return signals
        except Exception as e:
            self.logger.error(f"Error generating tax-aware signals: {e}")
            return {}

    def _calculate_tax_adjusted_return(self, symbol: str, raw_return: float) -> float:
        """Calculate return after considering tax implications"""
        holding_period = self.holding_periods.get(symbol, 0)
        # Apply lower tax rate for long-term holdings
        effective_tax_rate = self.tax_rate * 0.5 if holding_period > 365 else self.tax_rate
        return raw_return * (1 - effective_tax_rate)

    def _estimate_tax_impact(self, symbol: str) -> float:
        """Estimate tax impact of closing a position"""
        # Simple estimation - can be enhanced with actual P&L calculations
        return self.tax_rate if symbol in self.holding_periods else 0
