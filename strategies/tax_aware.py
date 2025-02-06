from typing import Dict, List
from models.client.profile import MockClientProfile
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

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

    async def process_market_data(self, market_data: Dict) -> Dict:
        """Process market data and return processed results"""
        try:
            processed_data = {}
            for symbol, data in market_data.items():
                processed_data[symbol] = {
                    'price': float(data.get('candles', [])[-1][4]),  # Close price
                    'volume': float(data.get('candles', [])[-1][5]),  # Volume
                    'returns': self._calculate_returns(data.get('candles', [])),
                    'timestamp': data.get('timestamp')
                }
                # Update holding period
                if symbol in self.holding_periods:
                    self.holding_periods[symbol] += 1
            return processed_data
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {}

    async def on_trade(self, trade_data: Dict) -> None:
        """Handle trade events and update strategy state"""
        try:
            symbol = trade_data.get('symbol')
            action = trade_data.get('action')
            
            if action == 'buy' and symbol not in self.holding_periods:
                # Start tracking holding period for new positions
                self.holding_periods[symbol] = 0
            elif action == 'sell':
                # Clear holding period on position exit
                self.holding_periods.pop(symbol, None)
                
            logger.info(f"Trade processed - Symbol: {symbol}, Action: {action}")
        except Exception as e:
            logger.error(f"Error processing trade: {e}")

    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        """Generate base trading signals considering tax implications"""
        signals = []
        try:
            for symbol, data in market_data.items():
                tax_adjusted_return = self._calculate_tax_adjusted_return(
                    symbol,
                    data.get('returns', 0)
                )
                
                # Generate signal only if tax-adjusted return is significant
                if abs(tax_adjusted_return) > 0.01:  # 1% threshold
                    signal = {
                        'symbol': symbol,
                        'direction': 'buy' if tax_adjusted_return > 0 else 'sell',
                        'strength': abs(tax_adjusted_return),
                        'size': 1.0,  # Base size, will be adjusted by risk manager
                        'metadata': {
                            'tax_impact': self._estimate_tax_impact(symbol),
                            'holding_period': self.holding_periods.get(symbol, 0)
                        }
                    }
                    signals.append(signal)
                    
            return signals
        except Exception as e:
            logger.error(f"Error generating base signals: {e}")
            return []

    def _calculate_returns(self, candles: list) -> float:
        """Calculate returns from candle data"""
        if not candles or len(candles) < 2:
            return 0.0
        try:
            current_price = float(candles[-1][4])  # Latest close price
            previous_price = float(candles[-2][4])  # Previous close price
            return (current_price - previous_price) / previous_price
        except (IndexError, ValueError) as e:
            logger.error(f"Error calculating returns: {e}")
            return 0.0

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
