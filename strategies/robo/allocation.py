from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ...strategies.base_strategy import BaseStrategy

class AssetAllocationStrategy(BaseStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.target_weights = {}
        self.risk_profile = config.get('risk_profile', 'moderate')
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        
    async def generate_signals(self) -> List[Dict]:
        current_weights = self._get_current_weights()
        rebalance_trades = []
        
        for asset, target in self.target_weights.items():
            current = current_weights.get(asset, 0.0)
            if abs(current - target) > self.rebalance_threshold:
                rebalance_trades.append({
                    'asset': asset,
                    'direction': 1 if target > current else -1,
                    'amount': abs(target - current),
                    'type': 'rebalance'
                })
                
        return rebalance_trades

    def _get_current_weights(self) -> Dict[str, float]:
        total_value = sum(self.positions.values())
        return {k: v/total_value for k, v in self.positions.items()}