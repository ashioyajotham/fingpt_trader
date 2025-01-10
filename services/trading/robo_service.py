from typing import Dict, List, Optional
import sys
from pathlib import Path
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.base_service import BaseService
from models.portfolio.rebalancing import Portfolio

class RoboService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.portfolio = Portfolio()
        self.rebalance_threshold = 0.05
        
    async def _setup(self) -> None:
        await self._load_portfolio()
        
    async def analyze_portfolio(self) -> Dict:
        """Analyze current portfolio state"""
        return {
            "weights": self.portfolio.get_current_weights(),
            "value": self.portfolio.get_portfolio_value(),
            "positions": self.portfolio.positions
        }
        
    async def generate_trades(self) -> List:
        """Generate rebalancing trades"""
        return self.portfolio.threshold_rebalance(self.rebalance_threshold)
        
    async def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices"""
        self.portfolio.update_prices(prices)