"""Tax-aware trading strategy implementation"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TaxAwareStrategy(BaseStrategy):
    """Tax-aware trading strategy with momentum and trend following"""
    
    def __init__(self, config: Dict = None, profile: Dict = None):
        super().__init__(config, profile)
        
        # Strategy parameters
        self.enabled = True
        self.momentum_window = 20  # Periods for momentum calculation
        self.trend_window = 50     # Periods for trend calculation
        self.min_holding_period = 30  # Minimum holding period in days
        self.tax_loss_threshold = -0.05  # 5% loss threshold
        self.position_history = {}  # Track position entry prices
        
        # Configure from dict if provided
        if config and 'strategies' in config and 'tax_aware' in config['strategies']:
            tax_config = config['strategies']['tax_aware']
            self.enabled = tax_config.get('enabled', True)
            self.min_holding_period = tax_config.get('min_holding_period', 30)
            self.tax_loss_threshold = tax_config.get('tax_loss_threshold', -0.05)

    async def _generate_base_signals(self, market_data: Dict) -> List[str]:
        """Generate trading signals based on momentum and trend"""
        signals = []
        
        if 'close' not in market_data:
            return signals
            
        price = market_data['close']
        sma20 = market_data.get('sma_20', price)
        sma50 = market_data.get('sma_50', price)
        
        # Momentum signal
        if price > sma20:
            signals.append('BUY')
        elif price < sma20:
            signals.append('SELL')
            
        # Trend signal
        if sma20 > sma50:
            signals.append('BUY')
        elif sma20 < sma50:
            signals.append('SELL')
            
        return signals

    async def process_market_data(self, data: Dict) -> Dict:
        """Process market data and calculate indicators"""
        processed = {}
        
        if 'price' in data:
            price = data['price']
            processed['close'] = price
            processed['sma_20'] = data.get('sma_20', price)
            processed['sma_50'] = data.get('sma_50', price)
            
            # Calculate returns if we have position history
            symbol = data.get('pair')
            if symbol in self.position_history:
                entry_price = self.position_history[symbol]['price']
                processed['unrealized_return'] = (price - entry_price) / entry_price
            
        return processed

    async def analyze(self, pair: str, current_price: float = None, price: float = None, 
                     position: float = 0, position_size: float = None,
                     holding_period: int = None, unrealized_pnl: float = None) -> Optional[str]:
        """Generate trading signals with tax-awareness
        
        Args:
            pair: Trading pair symbol
            current_price: Current market price (for compatibility with run_trader)
            price: Alternative price input (for compatibility with backtest)
            position: Current position size (legacy)
            position_size: Current position size (new format)
            holding_period: Days position has been held
            unrealized_pnl: Current unrealized profit/loss (optional)
            
        Returns:
            Optional[str]: Trading signal ('BUY', 'SELL', or None)
        """
        # Skip if strategy disabled
        if not self.enabled:
            return None
            
        # Use current_price if provided, otherwise fall back to price
        actual_price = current_price if current_price is not None else price
        
        # Handle both position and position_size parameters
        actual_position = position_size if position_size is not None else position
        
        # Prepare market data with unrealized PnL if provided
        market_data = await self.process_market_data({
            'pair': pair,
            'price': actual_price,
            'position': actual_position,
            'unrealized_return': unrealized_pnl if unrealized_pnl is not None else None
        })

        # Get base signals
        signals = await self._generate_base_signals(market_data)
        
        # Apply tax-aware logic for existing positions
        if actual_position > 0:
            # Check minimum holding period
            if holding_period is not None and holding_period < self.min_holding_period:
                return None  # Hold position if minimum period not met
            
            # Track position if not already tracked
            if pair not in self.position_history:
                self.position_history[pair] = {
                    'price': actual_price,
                    'timestamp': datetime.now()
                }
            
            # Check for tax-loss harvesting opportunity
            unrealized_return = market_data.get('unrealized_return', 0)
            if unrealized_return <= self.tax_loss_threshold:
                signals.append('SELL')  # Tax loss harvesting signal
                
        elif actual_position == 0 and 'BUY' in signals:
            # When entering new position, track it
            self.position_history[pair] = {
                'price': actual_price,
                'timestamp': datetime.now()
            }
            return 'BUY'
            
        # Return most common signal
        if not signals:
            return None
            
        return max(set(signals), key=signals.count)

    async def on_trade(self, trade_data: Dict) -> None:
        """Handle trade execution"""
        symbol = trade_data.get('symbol')
        side = trade_data.get('side')
        price = trade_data.get('price')
        
        if side == 'SELL':
            # Clear position history on sell
            self.position_history.pop(symbol, None)
        elif side == 'BUY':
            # Update position history on buy
            self.position_history[symbol] = {
                'price': price,
                'timestamp': datetime.now()
            }

    async def validate(self, **kwargs) -> bool:
        """Validate strategy parameters"""
        if not self.enabled:
            return False
            
        if self.min_holding_period < 0:
            return False
            
        if not -1 <= self.tax_loss_threshold <= 0:
            return False
            
        return True
