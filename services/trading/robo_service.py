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

    async def initialize(self):
        """Public initialization method"""
        await self._setup()

    async def _setup(self):
        """Implementation of abstract _setup method"""
        try:
            # Initialize portfolio with config
            await self.portfolio.initialize(self.config.get('portfolio', {}))
            logger.info("RoboService setup complete")
        except Exception as e:
            logger.error(f"RoboService setup failed: {e}")
            await self._cleanup()  # Ensure cleanup on failure
            raise

    async def cleanup(self):
        """Public cleanup method"""
        await self._cleanup()

    async def _cleanup(self):
        """Implementation of abstract _cleanup method"""
        try:
            if hasattr(self, 'portfolio'):
                # Just log for now since Portfolio doesn't have cleanup
                logger.info("Cleaning up portfolio resources")
            
            # Wait briefly for any pending operations
            await asyncio.sleep(0.1)
            logger.info("RoboService cleaned up")
        except Exception as e:
            logger.error(f"RoboService cleanup failed: {e}")
            raise