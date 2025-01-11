from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import aiohttp
import os

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
        self.api_key = os.getenv('ALPACA_API_KEY') or self.config.get('api_key')
        self.api_secret = os.getenv('ALPACA_API_SECRET') or self.config.get('api_secret')
        self.base_url = self.config.get('base_url', 'https://paper-api.alpaca.markets')
        self.session = None
        self.positions = {}
        self.orders = {}
        self.account_info = {}
        self.active = False

    async def _setup(self) -> None:
        """Initialize broker connection"""
        if not self.api_key or not self.api_secret:
            raise ValueError("Broker API credentials not set")
        self.session = aiohttp.ClientSession()
        self.active = True
        await self._validate_connection()
        await self._connect()
        await self._sync_positions()

    async def _cleanup(self) -> None:
        """Cleanup broker resources"""
        self.active = False
        if self.session:
            await self.session.close()
        self.positions.clear()
        self.orders.clear()

    async def _validate_connection(self) -> None:
        """Validate broker API connection"""
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        async with self.session.get(f"{self.base_url}/v2/account", headers=headers) as response:
            if response.status != 200:
                raise ConnectionError("Failed to connect to broker API")

    async def submit_order(self, order: Dict) -> str:
        """Submit new order to broker"""
        self._validate_order(order)
        order_id = await self._submit_to_broker(order)
        self.orders[order_id] = order
        return order_id

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