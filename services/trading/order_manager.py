from typing import Dict, Optional
from enum import Enum
from ...services.base_service import BaseService

class OrderStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderManager(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.active_orders = {}
        self.filled_orders = {}
        
    async def _setup(self) -> None:
        self.risk_limits = self.config.get('risk_limits', {})
        
    async def create_order(self, order_details: Dict) -> str:
        """Create and validate new order"""
        if not self._check_risk_limits(order_details):
            raise ValueError("Order exceeds risk limits")
        order_id = self._generate_order_id()
        self.active_orders[order_id] = {
            "status": OrderStatus.NEW,
            "details": order_details
        }
        return order_id

    async def update_order_status(self, order_id: str, 
                                status: OrderStatus, 
                                fill_info: Optional[Dict] = None) -> None:
        """Update order status and fill information"""
        if order_id not in self.active_orders:
            return
        
        order = self.active_orders[order_id]
        order["status"] = status
        
        if status == OrderStatus.FILLED:
            order["fill_info"] = fill_info
            self.filled_orders[order_id] = order
            del self.active_orders[order_id]