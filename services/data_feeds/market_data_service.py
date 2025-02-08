import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

import aiohttp
import ccxt.async_support as ccxt  # Use async version
import pandas as pd

from services.base_service import BaseService

logger = logging.getLogger(__name__)

class MarketDataService(BaseService):
    """
    Real-time market data service.
    
    Provides market data feeds from multiple exchanges with configurable
    update intervals and caching.
    
    Configuration:
        update_interval: Data refresh interval in seconds
        cache_expiry: Cache timeout in seconds
        exchanges: List of supported exchanges
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_SECRET_KEY")
        self.exchange = ccxt.binance(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                    "testnet": True,  # Use testnet
                },
                "urls": {
                    "api": {
                        "public": "https://testnet.binance.vision/api/v3",
                        "private": "https://testnet.binance.vision/api/v3",
                    }
                },
            }
        )
        self.last_update = datetime.now()
        self.rate_limit = self.config.get("rate_limits", {}).get(
            "requests_per_minute", 1200
        )
        self.update_interval = 1
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 5
        self.fallback_exchanges = ["kucoin", "huobi"]

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
            for attempt in range(self.max_retries):
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    quotes[symbol] = {
                        "price": ticker["last"],
                        "volume": ticker["baseVolume"],
                        "timestamp": ticker["timestamp"],
                    }
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(
                            f"Error fetching {symbol} after {self.max_retries} attempts"
                        )
                        # Try fallback exchanges
                        quotes[symbol] = await self._try_fallback_exchanges(symbol)
                    else:
                        await asyncio.sleep(self.retry_delay)

        return quotes

    async def _check_rate_limit(self) -> None:
        """Enforce rate limiting"""
        current_time = datetime.now()
        if (current_time - self.last_update).total_seconds() < self.update_interval:
            await asyncio.sleep(self.update_interval)
        self.last_update = current_time

class MarketDataFeed(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.pairs = config.get('pairs', ['BTCUSDT', 'ETHUSDT'])
        self.cache = {
            'candles': {},
            'trades': {},
            'orderbook': {},
            'ticker': {}
        }
        self.callbacks = []
        self.running = False
        self.data_handlers = []

    async def _setup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.running = True
            logger.info("Market data feed setup complete")
        except Exception as e:
            logger.error(f"Market data feed setup failed: {e}")
            raise

    async def _cleanup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.running = False
            self.cache.clear()
            self.data_handlers.clear()
            logger.info("Market data feed cleanup complete")
        except Exception as e:
            logger.error(f"Market data feed cleanup failed: {e}")
            raise

    async def _validate_pair(self, pair: str) -> bool:
        """Validate trading pair format"""
        return pair in self.pairs

    async def _handle_orderbook(self, data: Dict) -> None:
        """Process orderbook updates"""
        pair = data.get('symbol')
        if not await self._validate_pair(pair):
            return
            
        self.cache['orderbook'][pair] = {
            'data': data,
            'timestamp': datetime.now()
        }
        await self._notify_handlers('orderbook', pair, data)

    async def _handle_trades(self, data: Dict) -> None:
        """Process trade updates"""
        pair = data.get('symbol')
        if not await self._validate_pair(pair):
            return
            
        if pair not in self.cache['trades']:
            self.cache['trades'][pair] = []
            
        self.cache['trades'][pair].append({
            'data': data,
            'timestamp': datetime.now()
        })
        await self._notify_handlers('trades', pair, data)

    async def _notify_handlers(self, event_type: str, pair: str, data: Dict) -> None:
        """Notify registered handlers of updates"""
        for handler in self.data_handlers:
            try:
                await handler(event_type, pair, data)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    async def subscribe(self, handler) -> None:
        """Register a data handler"""
        self.data_handlers.append(handler)

    def get_latest(self, pair: str) -> Dict:
        """Get latest market data for pair"""
        return {
            'orderbook': self.cache['orderbook'].get(pair, {}),
            'trades': self.cache['trades'].get(pair, [])[-10:],
            'ticker': self.cache['ticker'].get(pair, {})
        }
