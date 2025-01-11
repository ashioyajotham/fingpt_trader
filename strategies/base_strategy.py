from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.positions = {}
        
    @abstractmethod
    async def process_market_data(self, data: Dict) -> None:
        """Process incoming market data"""
        pass

    @abstractmethod
    async def on_trade(self, trade: Dict) -> None:
        """Handle trade execution updates"""
        pass
    
    async def start(self) -> None:
        """Initialize and start strategy"""
        self.active = True
        self.last_update = datetime.now()
        await self._init_state()
    
    async def stop(self) -> None:
        """Stop strategy"""
        self.active = False
        await self._cleanup()
        
    async def update_position(self, symbol: str, quantity: float) -> None:
        """Update strategy position"""
        self.positions[symbol] = quantity
        
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions.copy()
        
    async def _init_state(self) -> None:
        """Initialize strategy state"""
        self.positions = {}
        self.signals = []
        
    async def _cleanup(self) -> None:
        """Cleanup strategy resources"""
        self.positions = {}
        self.signals = []
        
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal"""
        required = {'symbol', 'direction', 'strength'}
        return all(k in signal for k in required)