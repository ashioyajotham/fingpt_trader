from abc import ABC, abstractmethod
from typing import Dict, List

class BaseExchangeClient(ABC):
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    @abstractmethod
    async def initialize(self):
        """Initialize exchange client"""
        pass

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
        """Cleanup exchange client"""
        pass