from typing import Dict, List, Optional, Union
import pandas as pd
import asyncio
import ccxt.async_support as ccxt  # Use async version
import aiohttp
import os
from datetime import datetime, timedelta
from services.base_service import BaseService

class MarketDataService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_SECRET_KEY')
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.last_update = datetime.now()
        self.rate_limit = self.config.get('rate_limits', {}).get('requests_per_minute', 1200)
        self.update_interval = 1
        self.cache = {}

    async def start(self) -> None:
        """Start market data service"""
        await self._setup()
        
    async def stop(self) -> None:
        """Stop market data service"""
        await self._cleanup()
        
    async def _setup(self) -> None:
        """Initialize exchange connection"""
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not set")
        try:
            await self.exchange.load_markets()
            print("Connected to Binance API")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {str(e)}")

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.exchange:
            await self.exchange.close()
        self.cache.clear()

    async def get_realtime_quote(self, symbols: List[str]) -> Dict:
        """Get real-time market data"""
        await self._check_rate_limit()
        quotes = {}
        
        for symbol in symbols:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                quotes[symbol] = {
                    'price': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'timestamp': ticker['timestamp']
                }
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
                
        return quotes

    async def _check_rate_limit(self) -> None:
        """Enforce rate limiting"""
        current_time = datetime.now()
        if (current_time - self.last_update).total_seconds() < self.update_interval:
            await asyncio.sleep(self.update_interval)
        self.last_update = current_time