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