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
        
        Responsibilities:
        - Set initial portfolio positions (BTC, ETH, BNB)
        - Configure risk parameters
            * Maximum drawdown limits
            * Position size constraints
            * VaR (Value at Risk) limits
            * Leverage restrictions
        - Initialize portfolio state
        
        Raises:
            ValueError: If configuration is invalid
            Exception: If initialization fails
        """
        try:
            # Initialize with default portfolio positions
            initial_positions = {
                'BTCUSDT': 0.1,  # 0.1 BTC
                'ETHUSDT': 1.0,  # 1 ETH
                'BNBUSDT': 5.0   # 5 BNB
            }
            
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
            logger.info(f"Initial positions: {initial_positions}")
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
        
    async def execute_trade(self, signal_or_symbol, side=None, strength=None):
        """Execute a trade with proper validation and error handling"""
        try:
            # Support both calling conventions
            if isinstance(signal_or_symbol, dict):
                # Called with signal object
                signal = signal_or_symbol
                symbol = signal.get('symbol')
                side = signal.get('direction', signal.get('side', 'BUY'))
                strength = signal.get('strength', 0.5)
            else:
                # Called with individual parameters
                symbol = signal_or_symbol
                # side and strength already provided as params
                signal = {'symbol': symbol, 'side': side, 'strength': strength}
            
            # Calculate position size based on strength
            position_size = self._calculate_position_size(symbol, strength)

            # Get current price for the symbol
            current_price = 0
            if symbol in self.trading_pairs and 'binance' in self.exchange_clients:
                try:
                    ticker = await self.exchange_clients['binance'].get_ticker(symbol)
                    current_price = float(ticker['lastPrice'])
                except Exception as e:
                    logger.error(f"Failed to get price for {symbol}: {e}")
                    # Fallback prices
                    if symbol == 'BTCUSDT':
                        current_price = 100000
                    elif symbol == 'ETHUSDT':
                        current_price = 2500
                    elif symbol == 'BNBUSDT':
                        current_price = 600

            # NEW: Verify available balance before trading
            if side.upper() == 'BUY':
                # Check if we have enough quote currency (USDT)
                quote_currency = symbol.replace('BTC', '').replace('ETH', '').replace('BNB', '')
                required_balance = position_size * current_price * 1.01  # Add 1% buffer for fees
                
                # Get actual balance from exchange instead of internal tracking
                account_info = await self.exchange_clients['binance'].client.get_account()
                balances = account_info['balances']
                
                available_balance = 0
                for balance in balances:
                    if balance['asset'] == quote_currency:
                        available_balance = float(balance['free'])
                        break
                
                logger.info(f"Required balance: {required_balance} {quote_currency}, Available: {available_balance} {quote_currency}")
                
                if available_balance < required_balance:
                    logger.warning(f"Insufficient balance: {available_balance} < {required_balance} {quote_currency}")
                    return {'success': False, 'symbol': symbol, 'error': f"Insufficient balance for trade"}

            # Check minimum requirements
            min_notional = 15.0  # Minimum USD value
            min_qty = 0.00001 if symbol.startswith('BTC') else 0.001  # Default minimums
            if position_size * current_price < min_notional:
                logger.warning(f"Order too small: {position_size} {symbol} @ {current_price} = ${position_size * current_price:.2f} < ${min_notional}")
                return {'success': False, 'symbol': symbol, 'error': f"Order size too small (min ${min_notional})"}

            # Execute order based on side
            if side.upper() == 'BUY':
                return await self._execute_buy(symbol, position_size, current_price)
            elif side.upper() == 'SELL':
                return await self._execute_sell(symbol, position_size, current_price)
            else:
                return {'success': False, 'symbol': symbol, 'error': f"Invalid side: {side}"}
                
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return {'success': False, 'symbol': signal_or_symbol if isinstance(signal_or_symbol, str) else signal_or_symbol.get('symbol', 'UNKNOWN'), 'error': str(e)}

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
        """Calculate position size based on symbol and signal strength"""
        # Get balance and config values with defaults
        balance = self.get_balance()
        base_position = 0.01  # Base position size (1%)
        max_position = 0.05   # Maximum position size (5%)
        
        # Scale position size based on signal strength
        position_pct = base_position + ((max_position - base_position) * strength)
        position_value = balance * position_pct
        
        # Get current price to convert to quantity
        price = 0
        try:
            if 'binance' in self.exchange_clients:
                ticker = self.exchange_clients['binance'].get_ticker_sync(symbol)
                price = float(ticker['lastPrice'])
        except:
            # Fallback prices for testing
            if symbol == 'BTCUSDT':
                price = 100000
            elif symbol == 'ETHUSDT':
                price = 2500
            elif symbol == 'BNBUSDT':
                price = 600
        
        # Calculate quantity
        if price > 0:
            quantity = position_value / price
            
            # Apply minimum quantity check
            min_qty = 0.00001 if symbol.startswith('BTC') else 0.001
            if quantity < min_qty:
                quantity = min_qty
        else:
            quantity = 0
            
        return quantity

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
            if 'binance' in self.exchange_clients:
                # Get prices for all positions
                positions = self.portfolio.positions
                if not positions:
                    return
                    
                # Get current prices
                price_data = {}
                for symbol in positions.keys():
                    try:
                        ticker = await self.exchange_clients['binance'].get_ticker(symbol)
                        price = float(ticker.get('lastPrice', ticker.get('price', 0)))
                        if price > 0:
                            price_data[symbol] = {'price': price}
                    except Exception as e:
                        logger.warning(f"Failed to get price for {symbol}: {e}")
                
                # Update portfolio with new prices
                if price_data:
                    self.portfolio.update_prices(price_data)
                    logger.info(f"Updated prices for {len(price_data)} positions")
                    
                    # Record updated portfolio state
                    self.portfolio._record_portfolio_state()
        except Exception as e:
            logger.error(f"Error updating portfolio prices: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get a comprehensive summary of the current portfolio
        
        Returns:
            Dict containing:
                - cash: Available cash balance
                - positions: Current positions dict
                - total_value: Total portfolio value
                - returns: Dict of return metrics
                - risk: Dict of risk metrics
        """
        summary = {
            'cash': 0.0,
            'positions': {},
            'position_values': {},
            'total_value': 0.0,
            'returns': {
                'total': 0.0,
                'daily': 0.0
            },
            'risk': {
                'drawdown': 0.0,
                'volatility': 0.0
            }
        }
        
        if not hasattr(self, 'portfolio'):
            return summary
            
        try:
            # Get basic portfolio data
            summary['cash'] = self.portfolio.cash
            summary['positions'] = self.portfolio.positions.copy()
            
            # Calculate position values and total value
            total_value = summary['cash']
            for symbol, size in summary['positions'].items():
                price = self.portfolio.last_prices.get(symbol, 0)
                position_value = size * price
                summary['position_values'][symbol] = position_value
                total_value += position_value
            
            summary['total_value'] = total_value
            
            # Calculate returns if we have history
            if len(self.portfolio.portfolio_history) >= 2:
                initial_value = self.portfolio.portfolio_history[0]['total_value']
                summary['returns']['total'] = (total_value - initial_value) / initial_value
                
                # Calculate daily return if we have recent history
                if len(self.portfolio.portfolio_history) >= 2:
                    yesterday = self.portfolio.portfolio_history[-2]['total_value']
                    if yesterday > 0:
                        summary['returns']['daily'] = (total_value - yesterday) / yesterday
            
            # Calculate risk metrics
            if len(self.portfolio.portfolio_history) >= 2:
                values = [h['total_value'] for h in self.portfolio.portfolio_history]
                peak = max(values)
                summary['risk']['drawdown'] = (peak - total_value) / peak if peak > 0 else 0
                
                # Calculate volatility if we have enough data
                if len(values) >= 10:
                    returns = [(values[i] / values[i-1]) - 1 for i in range(1, len(values))]
                    summary['risk']['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
                    
            return summary
        
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return summary

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

    async def run(self):
        """Main run loop for the RoboService"""
        cycle_count = 0
        while True:
            try:
                # Add to run loop (periodic updates)
                if cycle_count % 10 == 0:  # Every 10 cycles
                    await self.sync_exchange_balances()

                # Perform other periodic tasks here
                await asyncio.sleep(1)
                cycle_count += 1
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                break