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
        self.last_price = {}  # Add this line to initialize price tracking
        
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
        self.strategies = {
            'tax_aware': TaxAwareStrategy(
                config=strategy_config.get('tax_aware', {}),
                profile=self.client_profile
            )
        }

    async def _setup(self):
        """Required implementation of abstract _setup method"""
        try:
            # Initialize portfolio with config
            await self.portfolio.initialize(self.config.get('portfolio', {}))
            
            # Initialize tax-aware strategy
            self.tax_aware_strategy = TaxAwareStrategy(self.config)
            logger.info("RoboService setup complete")
            
        except Exception as e:
            logger.error(f"RoboService setup failed: {e}")
            raise

    async def analyze_position(self, pair: str, price: float, position: float,
                             holding_days: int = 0) -> Optional[str]:
        """Get combined strategy signals for a position"""
        try:
            signals = []
            
            # Calculate PnL using stored last price or current price as fallback
            pnl = 0
            if pair in self.last_price:
                pnl = (price / self.last_price[pair] - 1)
            
            # Get tax strategy signal
            if hasattr(self, 'tax_aware_strategy'):
                tax_signal = await self.tax_aware_strategy.analyze(
                    pair=pair,
                    current_price=price,
                    position_size=position,
                    holding_period=holding_days,
                    unrealized_pnl=pnl
                )
                if tax_signal:
                    signals.append(tax_signal)

            # Store price for next PnL calculation
            self.last_price[pair] = price
            
            # Combine signals (prefer SELL over BUY)
            if 'SELL' in signals:
                return 'SELL'
            if all(s == 'BUY' for s in signals):
                return 'BUY'
            
            return None
            
        except Exception as e:
            logger.error(f"Position analysis failed: {e}")
            return None

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