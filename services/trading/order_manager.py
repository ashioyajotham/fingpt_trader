from ..base_service import BaseService
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

class OrderStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float] = None
    time_in_force: str = "DAY"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class OrderManager(BaseService):
    def _validate_config(self) -> None:
        required = ['max_position_size', 'risk_per_trade']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def initialize(self) -> None:
        self.orders = {}
        self.positions = {}
        self.pending_orders = set()
        self.broker_service = None
        self.logger = logging.getLogger(__name__)

    async def shutdown(self) -> None:
        # Cancel all pending orders
        for order_id in self.pending_orders.copy():
            await self.cancel_order(order_id)

    async def set_broker_service(self, broker_service) -> None:
        """Set the broker service for order execution"""
        self.broker_service = broker_service
        # Sync existing positions
        positions = await broker_service.get_positions()
        self.positions = {p['symbol']: p for p in positions}

    async def submit_order(self, order_request: OrderRequest) -> Dict:
        """Process and submit a new order"""
        # Validate order
        if not self._validate_order(order_request):
            return {'status': 'rejected', 'reason': 'validation_failed'}

        # Check risk limits
        if not self._check_risk_limits(order_request):
            return {'status': 'rejected', 'reason': 'risk_limits_exceeded'}

        # Create order record
        order_id = self._generate_order_id()
        order = {
            'id': order_id,
            'status': OrderStatus.VALIDATED,
            'request': order_request,
            'timestamp': datetime.now(),
            'fills': []
        }
        self.orders[order_id] = order
        self.pending_orders.add(order_id)

        try:
            # Submit to broker
            broker_response = await self.broker_service.place_order({
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': order_request.quantity,
                'type': order_request.order_type,
                'price': order_request.price,
                'time_in_force': order_request.time_in_force
            })

            # Update order status
            order['status'] = OrderStatus.SUBMITTED
            order['broker_order_id'] = broker_response['order_id']
            
            # Start monitoring order status
            asyncio.create_task(self._monitor_order(order_id))
            
            return {'status': 'submitted', 'order_id': order_id}

        except Exception as e:
            self.logger.error(f"Order submission failed: {str(e)}")
            order['status'] = OrderStatus.REJECTED
            self.pending_orders.remove(order_id)
            return {'status': 'rejected', 'reason': str(e)}

    def _validate_order(self, order_request: OrderRequest) -> bool:
        """Validate order parameters"""
        if order_request.quantity <= 0:
            return False
        if order_request.order_type == "limit" and not order_request.price:
            return False
        if order_request.side not in ["buy", "sell"]:
            return False
        return True

    def _check_risk_limits(self, order_request: OrderRequest) -> bool:
        """Check if order meets risk management criteria"""
        # Calculate position value
        position_value = order_request.quantity * (
            order_request.price or self._get_market_price(order_request.symbol)
        )

        # Check max position size
        if position_value > self.config['max_position_size']:
            return False

        # Check risk per trade
        risk = self._calculate_order_risk(order_request)
        if risk > self.config['risk_per_trade']:
            return False

        return True

    async def _monitor_order(self, order_id: str) -> None:
        """Monitor order status until filled or cancelled"""
        while order_id in self.pending_orders:
            order = self.orders[order_id]
            
            try:
                # Get latest order status from broker
                status = await self.broker_service.get_order_status(
                    order['broker_order_id']
                )
                
                # Update order status
                if status['status'] == 'filled':
                    await self._process_fill(order_id, status['fill_info'])
                elif status['status'] == 'cancelled':
                    self._process_cancellation(order_id)
                    break
                    
            except Exception as e:
                self.logger.error(f"Error monitoring order {order_id}: {str(e)}")
                
            await asyncio.sleep(1)  # Polling interval

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel a pending order"""
        if order_id not in self.pending_orders:
            return {'status': 'failed', 'reason': 'order_not_found'}

        order = self.orders[order_id]
        try:
            await self.broker_service.cancel_order(order['broker_order_id'])
            self._process_cancellation(order_id)
            return {'status': 'cancelled'}
        except Exception as e:
            return {'status': 'failed', 'reason': str(e)}

    def get_order_status(self, order_id: str) -> Dict:
        """Get current order status"""
        return self.orders.get(order_id, {'status': 'not_found'})
