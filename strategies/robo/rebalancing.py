from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
from strategies.base_strategy import BaseStrategy
from models.client.profile import MockClientProfile
import logging

logger = logging.getLogger(__name__)

class RebalanceType(Enum):
    TIME = "time"
    THRESHOLD = "threshold"
    TAX_LOSS = "tax_loss"

class RebalancingStrategy(BaseStrategy):
    def __init__(self, config: Dict, client_profile: Optional[MockClientProfile] = None):
        super().__init__(config)
        self.client_profile = client_profile
        self.config = config or {}
        self.last_rebalance = datetime.now()
        self.rebalance_frequency = timedelta(days=self.config.get("days", 90))
        self.threshold = self.config.get("threshold", 0.05)
        self.positions = {}

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        type: RebalanceType = RebalanceType.THRESHOLD,
    ) -> bool:
        if type == RebalanceType.TIME:
            return datetime.now() - self.last_rebalance > self.rebalance_frequency

        if type == RebalanceType.THRESHOLD:
            return any(
                abs(current_weights.get(k, 0) - v) > self.threshold
                for k, v in target_weights.items()
            )

        return False

    async def on_trade(self, trade_data: Dict) -> None:
        try:
            asset = trade_data['asset']
            amount = trade_data['amount']
            price = trade_data['price']
            self.positions[asset] = amount * price
            logger.info(f"Updated position for {asset}: {self.positions[asset]}")
        except Exception as e:
            logger.error(f"Error processing trade: {e}")

    async def process_market_data(self, market_data: Dict) -> None:
        try:
            for asset, data in market_data.items():
                if asset in self.positions:
                    self.positions[asset] *= float(data['price'])
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
