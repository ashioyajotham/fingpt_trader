from typing import Dict, List, Optional
from enum import Enum
from ...services.base_service import BaseService

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
        super().__init__(config)
        self.positions = {}
        self.orders = {}
        self.account_info = {}

    async def _setup(self) -> None:
        await self._connect()
        await self._sync_positions()
        
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