"""
Robo Advisory Service for Automated Trading

This service provides automated portfolio management with:
1. Client Profile Management
   - Risk tolerance scoring
   - Investment horizon tracking
   - Tax-aware trading preferences
   - ESG (Environmental, Social, Governance) preferences

2. Portfolio Management
   - Initial $10K allocation
   - Position sizing and rebalancing
   - Tax-loss harvesting
   - Risk-adjusted returns optimization

3. Trading Strategy Integration
   - Tax-aware trading execution
   - Portfolio rebalancing signals
   - Risk limit enforcement
   - Performance tracking

Architecture:
    RoboService
    ├── Portfolio (position management)
    ├── MockClientProfile (client preferences)
    └── TaxAwareStrategy (tax optimization)

Configuration:
    - initial_balance: Starting portfolio value
    - risk_score: Client risk tolerance (1-10)
    - investment_horizon: Target investment period
    - tax_rate: Client's tax bracket
    - position_limits: Min/max position sizes
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import requests
from aiohttp import ClientSession, TCPConnector

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.portfolio.rebalancing import Portfolio
from services.base_service import BaseService
from models.client.profile import MockClientProfile
from strategies.portfolio.tax_aware import TaxAwareStrategy

logger = logging.getLogger(__name__)

class RoboService(BaseService):
    """
    Automated portfolio management and trading service.
    
    Responsibilities:
        - Manage client profiles and preferences
        - Execute tax-aware trading strategies
        - Monitor portfolio performance
        - Enforce risk limits
        - Generate trading signals
    """
    
    def __init__(self, config: Dict):
        """
        Initialize RoboService with configuration.
        
        Args:
            config: Dictionary containing:
                - trading: Trading parameters
                - client_profile: Client preferences
                - strategies: Strategy configurations
                - risk: Risk management limits
        """
        super().__init__(config)
        
        # Initialize portfolio with default $10K or configured balance
        initial_balance = self.config.get('trading', {}).get('initial_balance', 10000.0)
        self.portfolio = Portfolio(initial_balance)
        
        # Setup client profile with risk preferences
        profile_config = config.get('client_profile', {})
        self.client_profile = MockClientProfile(
            risk_score=profile_config.get('risk_score', 5),       # Medium risk (1-10)
            investment_horizon=profile_config.get('investment_horizon', 365),  # 1 year
            constraints=profile_config.get('constraints', {}),     # Trading limits
            tax_rate=profile_config.get('tax_rate', 0.25),        # 25% tax rate
            esg_preferences=profile_config.get('esg_preferences', {})  # ESG settings
        )
        
        # Initialize tax-aware trading strategy
        strategy_config = config.get('strategies', {})
        self.tax_aware_strategy = TaxAwareStrategy(
            config=config,
            profile=self.client_profile.__dict__
        )

    async def setup(self, exchange_clients=None, trading_pairs=None, initial_balance=10000.0):
        """Public setup method for the RoboService"""
        # Store provided parameters
        self.exchange_clients = exchange_clients or {}
        self.trading_pairs = trading_pairs or ['BTCUSDT']
        self.initial_balance = initial_balance
        
        # Call the private setup method with the stored parameters
        await self.sync_exchange_balances()
        return await self._setup()

    async def _setup(self):
        """
        Initialize portfolio and risk management settings.
        """
        try:
            # Initialize with configuration-based portfolio positions
            config_positions = self.config.get('trading', {}).get('initial_positions', {})
            
            # If config doesn't specify positions, start with cash only
            if not config_positions:
                initial_positions = {}  # Start with cash only
                logger.info("Starting with cash-only portfolio based on configuration")
            else:
                initial_positions = config_positions
                logger.info(f"Using configured initial positions: {initial_positions}")
            
            # Get risk config with defaults
            risk_config = self.config.get('risk', {
                'max_drawdown': 0.10,
                'position_limit': 0.20,
                'var_limit': 0.02,
                'leverage_limit': 1.0
            })
            
            # Initialize portfolio with specific config
            portfolio_config = {
                'initial_balance': self.config.get('trading', {}).get('initial_balance', 10000.0),
                'positions': initial_positions,
                'position_limits': {
                    'max_position': risk_config.get('position_limit', 0.20),
                    'min_position': risk_config.get('min_position_size', 0.01)
                },
                'risk_limits': {
                    'max_drawdown': risk_config.get('max_drawdown', 0.10),
                    'max_leverage': risk_config.get('leverage_limit', 1.0),
                    'var_limit': risk_config.get('var_limit', 0.02)
                }
            }
            
            await self.portfolio.initialize(portfolio_config)
            logger.info(f"RoboService setup complete with initial balance: {portfolio_config['initial_balance']}")
            if initial_positions:
                logger.info(f"Initial positions: {initial_positions}")
            else:
                logger.info(f"Starting with cash-only portfolio: {portfolio_config['initial_balance']} USDT")
        except Exception as e:
            logger.error(f"RoboService setup failed: {e}")
            raise

    async def cleanup(self):
        """Public cleanup method"""
        await self._cleanup()

    async def _cleanup(self):
        """
        Clean shutdown of all RoboService components.
        
        Handles:
        - Portfolio state cleanup
        - Exchange connection closure
        - Resource deallocation
        - Graceful shutdown timing
        
        Raises:
            Exception: Logs error but continues cleanup
        """
        try:
            # Cleanup portfolio first
            if hasattr(self, 'portfolio'):
                await self.portfolio.cleanup()
            
            # Ensure exchange connections are closed
            if hasattr(self, 'exchange'):
                await self.exchange.cleanup()
                
            # Wait briefly for connections to fully close
            await asyncio.sleep(0.25)
            
            logger.info("RoboService cleaned up")
        except Exception as e:
            logger.error(f"RoboService cleanup failed: {e}")
            raise  # Re-raise to ensure proper error handling

    async def initialize_portfolio(self, config):
        """Initialize portfolio with configuration values"""
        try:
            # Create portfolio if it doesn't exist
            if not hasattr(self, 'portfolio'):
                initial_balance = config.get('trading', {}).get('initial_balance', 10000.0)
                self.portfolio = Portfolio(initial_balance)
                
            # Configure any existing positions
            initial_positions = config.get('trading', {}).get('initial_positions', {})
            for symbol, size in initial_positions.items():
                # Get current price for position
                price = 0
                if hasattr(self, 'exchange_clients') and 'binance' in self.exchange_clients:
                    try:
                        ticker = await self.exchange_clients['binance'].get_ticker(symbol)
                        price = float(ticker['lastPrice'])
                    except Exception as e:
                        logger.error(f"Failed to get price for {symbol}: {e}")
                        price = 0  # Default
                
                # Add position with entry price = current price if not already present
                if symbol not in self.portfolio.positions:
                    await self.portfolio.add_position(symbol, size, price, cost=0)  # No cost for initial positions
                    logger.info(f"Initialized position: {size} {symbol} @ {price}")
            
            # Initialize other portfolio settings
            await self.portfolio.initialize(config.get('portfolio', {}))
            logger.info(f"Portfolio initialized with balance: {self.portfolio.cash} USDT")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            return False

    async def analyze_position(self, pair: str, price: float, position: Dict = None) -> Optional[str]:
        """
        Generate trading signals based on multiple strategy inputs.
        
        Strategy Layers:
        1. Tax-Aware Trading
            - Tax-loss harvesting opportunities
            - Holding period optimization
            - Wash sale prevention
        
        2. Portfolio Optimization
            - Position size limits
            - Risk exposure management
            - Rebalancing triggers
        
        3. Risk Management
            - Drawdown protection
            - Leverage monitoring
            - Exposure limits
        
        Args:
            pair: Trading pair symbol (e.g., 'BTCUSDT')
            price: Current market price
            position: Optional position info with keys:
                     - size: Current position size
                     - entry_price: Entry price
                     - holding_period: Days held
        
        Returns:
            Optional[str]: Trading signal ('BUY', 'SELL', or None)
        
        Raises:
            Exception: Logs error and returns None on failure
        """
        try:
            # Convert float position to dict format
            if isinstance(position, (int, float)):
                position = {
                    'size': float(position),
                    'entry_price': price,
                    'holding_period': 0
                }
            # Get current position info if not provided
            elif position is None:
                position = self.portfolio.get_position(pair)

            # Extract position details with defaults            
            position_size = position.get('size', 0) if position else 0
            entry_price = position.get('entry_price', price) if position else price
            holding_period = position.get('holding_period', 0) if position else 0
            
            # Calculate unrealized PnL
            if position_size > 0:
                unrealized_pnl = (price - entry_price) / entry_price
            else:
                unrealized_pnl = 0

            # Collect signals from all strategies
            signals = []
            
            # Get tax-aware strategy signal
            tax_signal = await self.tax_aware_strategy.analyze(
                pair=pair,
                current_price=price, 
                position_size=position_size,
                holding_period=holding_period,
                unrealized_pnl=unrealized_pnl
            )
            if tax_signal:
                signals.append(tax_signal)

            # Determine final signal based on majority
            if not signals:
                return None
                
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            if buy_signals > sell_signals:
                return 'BUY'
            elif sell_signals > buy_signals:
                return 'SELL'
            
            return None

        except Exception as e:
            logger.error(f"Position analysis failed: {str(e)}")
            return None
        
    async def execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade based on signal"""
        try:
            # Extract signal details
            symbol = signal.get('symbol')
            side = signal.get('direction', 'BUY')
            price = signal.get('price', 0.0)
            
            # Get exchange requirements
            min_notional = 15.0  # Default min notional value for Binance
            min_qty = 0.0001     # Default min quantity for Binance
            
            # Get exchange-specific limits
            symbol_info = await self.get_exchange_minimum_requirements(symbol)
            if symbol_info:
                min_notional = symbol_info.get('min_notional', min_notional)
                min_qty = symbol_info.get('min_qty', min_qty)
                
            logger.info(f"Exchange requirements for {symbol}: min_qty={min_qty}, min_notional={min_notional}")
            
            # Rest of the method continues...
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"success": False, "error": str(e)}

    def _update_accumulated_amount(self, symbol: str, new_amount: float, new_strength: float):
        """Update accumulated amount using weighted strategy based on signal strength"""
        current_order = self.pending_orders[symbol]
        current_amount = current_order['amount']
        
        # Calculate total strength from previous signals
        total_strength = sum(s['strength'] for s in current_order['signals'])
        
        # Add new signal's strength
        total_strength += new_strength
        
        # Calculate weighted average
        weighted_amount = ((current_amount * (total_strength - new_strength)) + 
                         (new_amount * new_strength)) / total_strength
        
        return weighted_amount

    async def _execute_buy(self, symbol, position_size, current_price):
        """Execute a buy order and update portfolio"""
        logger.info(f"Executing BUY for {symbol} with quantity {position_size}")
        try:
            if 'binance' not in self.exchange_clients:
                return {'success': False, 'symbol': symbol, 'error': "Binance client not available"}
                
            # Execute order
            result = await self.exchange_clients['binance'].create_market_buy_order(symbol, position_size)
            
            # Update portfolio if order successful
            if result:
                # Get actual execution price and quantity
                fills = result.get('details', {}).get('fills', [{'price': current_price}])
                actual_price = float(fills[0]['price']) if fills else current_price
                
                # Calculate total cost including commission
                commission = result.get('details', {}).get('cummulativeQuoteQty', position_size * actual_price)
                
                # Update portfolio tracking
                if hasattr(self, 'portfolio'):
                    portfolio_updated = await self.portfolio.add_position(symbol, position_size, actual_price, commission)
                    if portfolio_updated:
                        logger.info(f"Portfolio updated: Added {position_size} {symbol} @ {actual_price}")
                    else:
                        logger.warning(f"Portfolio update failed for {symbol}")
                
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                return {'success': True, 'symbol': symbol, 'side': 'BUY', 
                        'quantity': position_size, 'price': actual_price, 
                        'details': result}
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return {'success': False, 'symbol': symbol, 'error': str(e)}

    async def _execute_sell(self, symbol, position_size, current_price):
        """Execute a sell order and update portfolio"""
        logger.info(f"Executing SELL for {symbol} with quantity {position_size}")
        try:
            if 'binance' not in self.exchange_clients:
                return {'success': False, 'symbol': symbol, 'error': "Binance client not available"}
                
            # Execute order
            result = await self.exchange_clients['binance'].create_market_sell_order(symbol, position_size)
            
            # Update portfolio if order successful
            if result:
                # Get actual execution price
                fills = result.get('details', {}).get('fills', [{'price': current_price}])
                actual_price = float(fills[0]['price']) if fills else current_price
                
                # Update portfolio tracking
                if hasattr(self, 'portfolio'):
                    portfolio_updated = await self.portfolio.reduce_position(symbol, position_size, actual_price)
                    if portfolio_updated:
                        logger.info(f"Portfolio updated: Removed {position_size} {symbol} @ {actual_price}")
                    else:
                        logger.warning(f"Portfolio update failed for {symbol}")
                
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                return {'success': True, 'symbol': symbol, 'side': 'SELL', 
                        'quantity': position_size, 'price': actual_price, 
                        'details': result}
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return {'success': False, 'symbol': symbol, 'error': str(e)}

    def _calculate_performance_metrics(self):
        """Calculate and log portfolio performance metrics"""
        if not hasattr(self, 'portfolio') or not self.portfolio.portfolio_history:
            return
            
        try:
            # Calculate metrics only if we have history
            if len(self.portfolio.portfolio_history) >= 2:
                # Calculate return
                current_value = self.portfolio.portfolio_history[-1]['total_value']
                initial_value = self.portfolio.portfolio_history[0]['total_value']
                total_return = (current_value - initial_value) / initial_value
                
                # Calculate drawdown
                peak_value = max(h['total_value'] for h in self.portfolio.portfolio_history)
                current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
                
                logger.info(f"Portfolio metrics - Value: ${current_value:.2f}, Return: {total_return:.2%}, Drawdown: {current_drawdown:.2%}")
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")

    def _calculate_position_size(self, symbol, strength):
        """Calculate position size based on symbol, signal strength and market regime"""
        # Get balance and config values with defaults
        balance = self.get_balance()
        base_position = 0.01  # Base position size (1%)
        max_position = 0.05   # Maximum position size (5%)
        
        # Detect current market regime if available
        market_regime_factor = 1.0  # Default factor
        if hasattr(self, 'market_regime_detector'):
            # Get latest market data
            market_data = {}
            if hasattr(self, 'market_data_service'):
                market_data = self.market_data_service.get_market_data(symbol)
            
            # Detect regime and adjust position sizing
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Adjust position size based on regime
            if regime == "high_volatility":
                market_regime_factor = 0.7
            elif regime == "low_volatility":
                market_regime_factor = 1.2
            elif regime == "trending":
                market_regime_factor = 1.1
            elif regime == "crisis":
                market_regime_factor = 0.3
        
        # Scale position size based on signal strength
        position_pct = base_position + ((max_position - base_position) * strength)
        
        # Apply market regime factor
        position_pct *= market_regime_factor
        
        position_value = balance * position_pct
        
        # Get current price to convert to quantity
        price = 0
        try:
            if 'binance' in self.exchange_clients:
                # async method in an async context or fetch price from market data service
                price = self.market_data_service.get_latest_price(symbol) if hasattr(self, 'market_data_service') else 0
                
                # If market data service didn't provide a price, try fallback
                if price <= 0 and hasattr(self, 'portfolio') and hasattr(self.portfolio, 'last_prices'):
                    price = self.portfolio.last_prices.get(symbol, 0)
        except Exception as e:
            logger.error(f"Error getting price for position calculation: {e}")
            return 0  # Return 0 quantity if no price is available
        
        # Calculate quantity
        if price > 0:
            quantity = position_value / price
            
            # Apply minimum quantity check
            min_qty = 0.00001 if symbol.startswith('BTC') else 0.001
            if quantity < min_qty:
                quantity = min_qty
        else:
            logger.warning(f"Invalid price (0) for {symbol}, cannot calculate position size")
            quantity = 0
                
        return quantity

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a trading pair
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT')
            
        Returns:
            float: Current price or 0 if unavailable
        """
        if 'binance' not in self.exchange_clients:
            logger.error("Binance client not available")
            return 0.0
            
        try:
            ticker = await self.exchange_clients['binance'].get_ticker(symbol)
            if ticker and 'lastPrice' in ticker:
                return float(ticker['lastPrice'])
            logger.warning(f"No price data available from exchange for {symbol}")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_positions(self):
        """
        Get current portfolio positions.
        
        Returns:
            dict: Current positions with symbol as key and position details as value
        """
        if hasattr(self, 'portfolio'):
            return self.portfolio.positions
        return {}
        
    def get_balance(self):
        """
        Get available balance.
        
        Returns:
            float: Available balance in base currency
        """
        if hasattr(self, 'portfolio'):
            return self.portfolio.cash  # Use 'cash' instead of 'balance'
        return 0.0

    async def update_portfolio_prices(self):
        """Update portfolio with latest market prices"""
        if not hasattr(self, 'portfolio'):
            return
        
        try:
            # Get positions
            positions = self.portfolio.positions
            if not positions:
                return
                
            # Get current prices
            price_data = {}
        
            # Get prices from exchange clients
            if hasattr(self, 'exchange_clients') and 'binance' in self.exchange_clients:
                for symbol in positions.keys():
                    try:
                        ticker = await self.exchange_clients['binance'].get_ticker(symbol)
                        price = float(ticker.get('lastPrice', ticker.get('price', 0)))
                        if price > 0:
                            price_data[symbol] = {'price': price}
                    except Exception as e:
                        logger.warning(f"Failed to get price for {symbol}: {e}")
        
            # Update portfolio with whatever prices we could get
            if price_data:
                self.portfolio.update_prices(price_data)
                logger.info(f"Updated prices for {len(price_data)} positions")
                
                # Record updated portfolio state
                self.portfolio._record_portfolio_state()
            else:
                logger.warning("No price data available to update portfolio")
    
        except Exception as e:
            logger.error(f"Error updating portfolio prices: {e}")

    def get_portfolio_summary(self):
        """Get a summary of the portfolio for performance tracking"""
        if not hasattr(self, 'portfolio'):
            return {'total_value': 0, 'cash': 0, 'positions': {}}
        
        # Calculate total portfolio value
        total_value = self.portfolio.cash
        positions_data = {}
        
        for symbol, size in self.portfolio.positions.items():
            # Get current price
            price = self.portfolio.last_prices.get(symbol, 0)
            value = size * price
            total_value += value
            
            # Get entry price if available
            entry_price = 0
            if hasattr(self.portfolio, 'position_entries') and symbol in self.portfolio.position_entries:
                entry_price = self.portfolio.position_entries[symbol]
            
            # Calculate P&L if entry price is available
            pnl = 0
            if entry_price > 0 and price > 0:
                pnl = size * (price - entry_price)
            
            positions_data[symbol] = {
                'size': size,
                'price': price,
                'value': value,
                'entry_price': entry_price,
                'pnl': pnl
            }
        
        return {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'positions': positions_data,
            'timestamp': datetime.now()
        }

    async def sync_exchange_balances(self):
        """Synchronize internal portfolio tracking with actual exchange balances"""
        try:
            if 'binance' not in self.exchange_clients:
                logger.error("Binance client not available")
                return False
                
            # Get account information from exchange
            account_info = await self.exchange_clients['binance'].client.get_account()
            
            # Clear existing cash balance and update with actual exchange balance
            if hasattr(self, 'portfolio'):
                # Update cash (USDT) balance
                for balance in account_info['balances']:
                    if balance['asset'] == 'USDT':
                        self.portfolio.cash = float(balance['free'])
                        logger.info(f"Updated portfolio cash to {self.portfolio.cash} USDT from exchange")
                    
                    # Update crypto positions
                    asset = balance['asset']
                    if asset in ['BTC', 'ETH', 'BNB'] and float(balance['free']) > 0:
                        symbol = f"{asset}USDT"
                        self.portfolio.positions[symbol] = float(balance['free'])
                        logger.info(f"Updated portfolio position: {symbol} = {float(balance['free'])}")
                
                return True
        except Exception as e:
            logger.error(f"Error syncing exchange balances: {e}")
            return False

    async def get_exchange_minimum_requirements(self, symbol: str) -> Dict:
        """
        Dynamically fetch minimum order requirements from the exchange
        
        Returns:
            Dict with min_qty, min_notional, and other requirements
        """
        try:
            if 'binance' not in self.exchange_clients:
                logger.warning("Binance client not initialized, using default minimums")
                return {
                    'min_qty': 0.00001,  # Default minimum quantity
                    'min_notional': 15.0  # Default minimum notional value
                }
            
            # Get exchange info from Binance
            exchange_info = await self.exchange_clients['binance'].client.get_exchange_info()
            
            # Find the symbol info
            symbol_info = None
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                logger.warning(f"Symbol {symbol} not found in exchange info, using defaults")
                return {
                    'min_qty': 0.00001,
                    'min_notional': 15.0
                }
            
            # Extract minimum quantity and notional value
            min_qty = None
            min_notional = None
            
            # Process filters to find minQty and minNotional
            for f in symbol_info.get('filters', []):
                if f.get('filterType') == 'LOT_SIZE':
                    min_qty = float(f.get('minQty', 0.00001))
                elif f.get('filterType') == 'MIN_NOTIONAL':
                    min_notional = float(f.get('minNotional', 15.0))
            
            # Use defaults if not found
            if min_qty is None:
                min_qty = 0.00001
            if min_notional is None:
                min_notional = 15.0
                
            logger.info(f"Exchange requirements for {symbol}: min_qty={min_qty}, min_notional={min_notional}")
            return {
                'min_qty': min_qty,
                'min_notional': min_notional
            }
        except Exception as e:
            logger.error(f"Error fetching exchange minimums: {e}")
            return {
                'min_qty': 0.00001,
                'min_notional': 15.0
            }

    async def cleanup_expired_orders(self):
        """Remove pending orders that have expired"""
        if not hasattr(self, 'pending_orders'):
            return
        
        now = datetime.now()
        # Get expiration time from config (default 24 hours)
        expiration_seconds = self.config.get('trading', {}).get(
            'minimum_size_handling', {}).get('expiration_time', 86400)
    
        expired_symbols = []
        for symbol, order in self.pending_orders.items():
            if (now - order['created_at']).total_seconds() > expiration_seconds:
                expired_symbols.append(symbol)
                logger.warning(f"Order accumulation for {symbol} expired after {expiration_seconds/3600:.1f} hours")
    
        # Remove expired orders
        for symbol in expired_symbols:
            del self.pending_orders[symbol]
        
        if expired_symbols:
            logger.info(f"Cleaned up {len(expired_symbols)} expired order accumulations")

    async def update_portfolio_with_trade_result(self, result):
        """Update portfolio with executed trade result"""
        if not result or not result.get('success', False) or not hasattr(self, 'portfolio'):
            return
        
        symbol = result.get('symbol')
        side = result.get('side')
        quantity = result.get('quantity', 0)
        price = result.get('price', 0)
        
        try:
            if side.upper() == 'BUY':
                # Double-check that portfolio was updated correctly
                if symbol not in self.portfolio.positions or \
                   self.portfolio.positions[symbol] < quantity:
                    # Portfolio wasn't updated, force update
                    await self.portfolio.add_position(symbol, quantity, price)
                    logger.info(f"Portfolio manually updated: Added {quantity} {symbol} @ {price}")
            elif side.upper() == 'SELL':
                # For sell orders, ensure position was reduced
                if symbol in self.portfolio.positions:
                    current_pos = self.portfolio.positions[symbol]
                    if abs(current_pos - (result.get('original_position', 0) - quantity)) > 1e-8:
                        # Position wasn't updated correctly
                        await self.portfolio.reduce_position(symbol, quantity, price)
                        logger.info(f"Portfolio manually updated: Reduced {quantity} {symbol} @ {price}")
        
            # Record trade in history for better tracking
            self._record_trade_history(symbol, side, quantity, price)
            
            # Force update portfolio prices and recalculate state
            await self.update_portfolio_prices()
        except Exception as e:
            logger.error(f"Error updating portfolio with trade: {e}")

    def _record_trade_history(self, symbol, side, quantity, price):
        """Record trade in history for performance tracking"""
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        
        # Calculate cost/proceeds
        value = quantity * price
        
        # Record trade with full details
        self.trade_history.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': value,
            'timestamp': datetime.now(),
            'balance_after': self.portfolio.cash if hasattr(self, 'portfolio') else 0,
            'portfolio_value': self.portfolio.total_value() if hasattr(self, 'portfolio') else 0
        })
        
        # Limit history size
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        logger.info(f"Trade recorded: {side} {quantity} {symbol} @ {price} = ${value:.2f}")

    async def run(self):
        """Main run loop for the RoboService"""
        cycle_count = 0
        while True:
            try:
                # Add to run loop (periodic updates)
                if cycle_count % 10 == 0:  # Every 10 cycles
                    await self.sync_exchange_balances()  # Make sure this is awaited

                # Perform other periodic tasks here
                await asyncio.sleep(1)
                cycle_count += 1
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                break