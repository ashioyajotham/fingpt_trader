from abc import ABC, abstractmethod
from typing import Dict, List

class BaseExchangeClient(ABC):
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for symbol"""
        pass
        
    @abstractmethod
    async def place_order(self, order: Dict) -> Dict:
        """Place an order"""
        pass
        
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass