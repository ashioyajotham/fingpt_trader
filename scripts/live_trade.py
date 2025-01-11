import asyncio
import sys
from pathlib import Path
from typing import Dict


from services.trading.broker_service import BrokerService
from strategies.base_strategy import BaseStrategy
from utils.config import ConfigManager

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)


class LiveTrader:
    def __init__(self, config: Dict):
        self.config = config
        self.broker = BrokerService(config.get("broker"))
        self.running = False

    async def start(self, strategy: BaseStrategy):
        """Start live trading"""
        self.running = True
        await self.broker.start()

        while self.running:
            signals = await strategy.generate_signals()
            await self._execute_signals(signals)
            await asyncio.sleep(self.config.get("interval", 60))
