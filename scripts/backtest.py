import asyncio
import pandas as pd
from typing import Dict, List

import sys
from pathlib import Path
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from strategies.base_strategy import BaseStrategy
from utils.config import ConfigManager

class Backtester:
    def __init__(self, config: Dict):
        self.config = config
        self.data = {}
        self.results = []
        self.stats = {}
        
    async def run(self, strategy: BaseStrategy, start_date: str, end_date: str):
        """Run backtest"""
        await self._load_data(start_date, end_date)
        await self._simulate_trading(strategy)
        self._calculate_stats()
        return self.stats

    async def _load_data(self, start_date: str, end_date: str):
        """Load historical data"""
        self.data = {
            'market': pd.DataFrame(),
            'sentiment': pd.DataFrame(),
            'fundamentals': pd.DataFrame()
        }