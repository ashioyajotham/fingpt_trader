from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class RebalanceType(Enum):
    TIME = "time"
    THRESHOLD = "threshold"
    TAX_LOSS = "tax_loss"


class RebalancingStrategy:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.last_rebalance = datetime.now()
        self.rebalance_frequency = timedelta(days=self.config.get("days", 90))
        self.threshold = self.config.get("threshold", 0.05)

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
