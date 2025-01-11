from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import aiohttp
import os
import asyncio
import ccxt

from pathlib import Path
import sys

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
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.max_retries = 3
        self.retry_delay = 5
        self.positions = {}
        self.orders = {}
        self.account_info = {}
        self.active = False

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
                symbol=order['symbol'],
                type='market',
                side=order['side'],
                amount=order['amount']
            )
            self.orders[response['id']] = response
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