from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import asyncio
from dataclasses import dataclass
from ..base_service import BaseService
import aiohttp

@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit'
    price: Optional[float] = None
    status: str = 'pending'
    timestamp: datetime = None
    
class BrokerService(BaseService):
    def _validate_config(self) -> None:
        required = ['api_key', 'api_secret', 'base_url']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def initialize(self) -> None:
        self.session = None
        self.orders = {}
        self.positions = {}

    async def shutdown(self) -> None:
        if self.session:
            await self.session.close()

    async def _ensure_session(self) -> None:
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config['api_key']}"}
            )

    async def place_order(self, order: Dict) -> Dict:
        await self._ensure_session()
        async with self.session.post(
            f"{self.config['base_url']}/orders",
            json=order
        ) as response:
            result = await response.json()
            if response.status == 200:
                self.orders[result['order_id']] = result
            return result

    async def get_positions(self) -> List[Dict]:
        await self._ensure_session()
        async with self.session.get(
            f"{self.config['base_url']}/positions"
        ) as response:
            positions = await response.json()
            self.positions = {p['symbol']: p for p in positions}
            return positions