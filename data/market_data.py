from typing import Dict, List
import aiohttp
import asyncio
import pandas as pd
from abc import ABC, abstractmethod

class MarketDataSource(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.cache = {}
        
    async def connect(self):
        self.session = aiohttp.ClientSession()
        
    async def disconnect(self):
        if self.session:
            await self.session.close()
            
    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        pass
    
    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict:
        pass

class AlphaVantageAPI(MarketDataSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        
    async def get_price(self, symbol: str) -> float:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()
            return float(data["Global Quote"]["05. price"])