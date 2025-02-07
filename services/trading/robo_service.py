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
from strategies.tax_aware import TaxAwareStrategy

logger = logging.getLogger(__name__)

class RoboService(BaseService):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.portfolio = Portfolio()  # For managing trading positions
        
        # Initialize client profile with default values or from config
        profile_config = config.get('client_profile', {})
        self.client_profile = MockClientProfile(
            risk_score=profile_config.get('risk_score', 5),  # Medium risk (1-10)
            investment_horizon=profile_config.get('investment_horizon', 365),  # 1 year in days
            constraints=profile_config.get('constraints', {}),  # Trading constraints
            tax_rate=profile_config.get('tax_rate', 0.25),  # 25% default tax rate
            esg_preferences=profile_config.get('esg_preferences', {})  # ESG preferences
        )
        
        # Initialize strategies with proper configuration
        strategy_config = config.get('strategies', {})
        self.tax_aware_strategy = TaxAwareStrategy(
            config=config,  # Pass full config
            profile=self.client_profile.__dict__  # Convert profile to dict
        )

    async def _setup(self):
        """Required implementation of abstract _setup method"""
        try:
            # Initialize portfolio with config
            await self.portfolio.initialize(self.config.get('portfolio', {}))
            logger.info("RoboService setup complete")
        except Exception as e:
            logger.error(f"RoboService setup failed: {e}")
            raise

    async def cleanup(self):
        """Public cleanup method"""
        await self._cleanup()

    async def _cleanup(self):
        """Required implementation of abstract _cleanup method"""
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
        Analyze position and generate trading signals based on strategies
        
        Args:
            pair: Trading pair symbol (e.g. 'BTCUSDT') 
            price: Current market price
            position: Optional position info dictionary or float amount, will be fetched if not provided
            
        Returns:
            Optional[str]: Trading signal ('BUY', 'SELL', or None)
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