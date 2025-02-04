import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from strategies.base_strategy import BaseStrategy
from models.client.profile import MockClientProfile

logger = logging.getLogger(__name__)

@dataclass
class AllocationConfig:
    risk_profile: str
    rebalance_threshold: float
    min_position_size: float
    max_position_size: float
    max_leverage: float

class AssetAllocationStrategy(BaseStrategy):
    def __init__(self, config: Dict, client_profile: MockClientProfile):
        super().__init__(config)
        self.client_profile = client_profile
        self.positions = {}
        self.config = AllocationConfig(
            risk_profile=config.get("risk_profile", "moderate"),
            rebalance_threshold=config.get("rebalance_threshold", 0.05),
            min_position_size=client_profile.constraints.get('min_position_size', 0.01),
            max_position_size=client_profile.constraints.get('max_position_size', 0.1),
            max_leverage=client_profile.constraints.get('max_leverage', 1.0)
        )
        self.target_weights = self._calculate_target_weights()

    async def on_trade(self, trade_data: Dict) -> None:
        """
        Handle trade execution updates
        
        Args:
            trade_data (Dict): Information about executed trade including asset,
                             amount, price and direction
        """
        try:
            asset = trade_data['asset']
            amount = trade_data['amount']
            price = trade_data['price']
            direction = trade_data['direction']
            
            # Update position value
            current_value = self.positions.get(asset, 0)
            new_value = current_value + (amount * price * direction)
            self.positions[asset] = new_value
            
            logger.info(f"Updated position for {asset}: {self.positions[asset]:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def process_market_data(self, market_data: Dict) -> None:
        """
        Process new market data to update strategy state
        
        Args:
            market_data (Dict): Current market prices and volumes
        """
        try:
            # Update position values based on current prices
            for asset, data in market_data.items():
                if asset in self.positions:
                    price = float(data['price'])
                    self.positions[asset] = self.positions[asset] * price
            
            logger.debug(f"Updated positions with new market data")
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")

    def _calculate_target_weights(self) -> Dict[str, float]:
        """Calculate target weights based on risk profile and constraints"""
        risk_score = self.client_profile.risk_score
        weights = {}
        
        if risk_score <= 3:  # Conservative
            weights = {
                'BTCUSDT': 0.1,
                'ETHUSDT': 0.2,
                'USDTUSD': 0.7
            }
        elif risk_score <= 7:  # Moderate
            weights = {
                'BTCUSDT': 0.3,
                'ETHUSDT': 0.3,
                'USDTUSD': 0.4
            }
        else:  # Aggressive
            weights = {
                'BTCUSDT': 0.4,
                'ETHUSDT': 0.4,
                'USDTUSD': 0.2
            }
        
        return weights

    async def generate_signals(self) -> List[Dict]:
        """Generate rebalancing signals based on current positions"""
        try:
            current_weights = self._get_current_weights()
            rebalance_trades = []

            for asset, target in self.target_weights.items():
                current = current_weights.get(asset, 0.0)
                deviation = abs(current - target)
                
                if deviation > self.config.rebalance_threshold:
                    # Apply position size constraints
                    amount = min(
                        abs(target - current),
                        self.config.max_position_size
                    )
                    amount = max(amount, self.config.min_position_size)
                    
                    rebalance_trades.append({
                        "asset": asset,
                        "direction": 1 if target > current else -1,
                        "amount": amount,
                        "type": "rebalance",
                        "reason": f"Rebalancing {asset}: Current {current:.2%} Target {target:.2%}"
                    })
                    
            logger.info(f"Generated {len(rebalance_trades)} rebalancing signals")
            return rebalance_trades
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def _get_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = sum(self.positions.values())
        if total_value == 0:
            return {k: 0.0 for k in self.target_weights.keys()}
        return {k: v / total_value for k, v in self.positions.items()}
