import asyncio
import os
import sys
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
import ccxt
from models.portfolio.risk import CircuitBreaker, MarketRegimeDetector, MarketRegime

# Configure logger
logger = logging.getLogger(__name__)

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class BrokerService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.exchange = ccxt.binance(
            {
                "apiKey": os.getenv("BINANCE_API_KEY"),
                "secret": os.getenv("BINANCE_SECRET_KEY"),
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        self.max_retries = 3
        self.retry_delay = 5
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, Dict] = {}
        self.account_info: Dict[str, Any] = {}
        self.active = False

        # Initialize risk management components
        self.circuit_breaker = CircuitBreaker(config.get('risk', {}).get('thresholds', {}))
        self.regime_detector = MarketRegimeDetector()

        # Set up logging level from config
        log_level = config.get('logging', {}).get('level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

    async def _setup(self) -> None:
        """Initialize broker connection"""
        try:
            # Test connection
            await self._validate_connection()
            print("Successfully connected to Binance API")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {str(e)}")

    async def _cleanup(self) -> None:
        """Cleanup broker resources"""
        self.active = False
        self.positions.clear()
        self.orders.clear()

    async def _validate_connection(self) -> None:
        """Test API connection"""
        try:
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            print(f"Connected to Binance - Total BTC Value: {balance['total']['BTC']}")
        except Exception as e:
            raise ConnectionError(f"API Error: {str(e)}")

    async def submit_order(self, order: Dict) -> Dict:
        """Submit order to exchange"""
        try:
            response = self.exchange.create_order(
                symbol=order["symbol"],
                type="market",
                side=order["side"],
                amount=order["amount"],
            )
            self.orders[response["id"]] = response
            return response
        except Exception as e:
            raise RuntimeError(f"Order submission failed: {str(e)}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        if order_id not in self.orders:
            return False
        success = await self._cancel_at_broker(order_id)
        if success:
            del self.orders[order_id]
        return success

    async def get_positions(self) -> Dict:
        """Get current positions"""
        return self.positions

    async def get_account_info(self) -> Dict:
        """Get account information"""
        return self.account_info

    async def execute_order(self, order: Dict) -> bool:
        """Execute order with market risk checks"""
        try:
            # Get current market data
            market_data = await self._fetch_market_data(order['symbol'])
            
            # Check circuit breaker conditions
            if self.circuit_breaker.check_conditions(market_data):
                logger.warning(f"Circuit breaker triggered for {order['symbol']}")
                return False
            
            # Check market regime
            regime = self.regime_detector.detect_regime(market_data)
            if regime in [MarketRegime.CRISIS, MarketRegime.STRESS]:
                logger.warning(f"Order rejected due to market regime: {regime}")
                return False
                
            # Proceed with order execution if checks pass
            return await self._submit_order(order)
            
        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return False
            
    async def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch market data for risk checks"""
        pass
