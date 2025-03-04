import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd

from services.base_service import BaseService
from ..exchanges.binance import BinanceClient
import logging
logger = logging.getLogger(__name__)

class MarketDataService(BaseService):
    """
    Real-time market data service with news integration
    
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
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.exchange = None  # Will be set during _setup
        self.last_update = datetime.now()
        self.rate_limit = self.config.get("rate_limits", {}).get(
            "requests_per_minute", 1200
        )
        self.update_interval = 1
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 5
        self.fallback_exchanges = ["kucoin", "huobi"]
        
        # Add news correlation tracking
        self.news_impacts = {}
        self.price_events = {}
        self.correlation_window = timedelta(hours=24)
        self.min_correlation_samples = 10
        
        # Event detection thresholds
        self.volatility_threshold = 0.02  # 2% price move
        self.volume_threshold = 2.0  # 2x average volume
        
        # Cache settings
        self.cache_ttl = timedelta(minutes=5)
        self.price_history = {}
        self.volume_history = {}

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
            # Use singleton instance instead of creating new connection
            self.exchange = await BinanceClient.get_instance({
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'test_mode': True
            })
            logger.info("Using shared Binance client instance")
            
            # Initialize event tracking
            for pair in self.config.get('pairs', []):
                self.price_events[pair] = []
                self.price_history[pair] = []
                self.volume_history[pair] = []
            
            logger.info("Market data service initialized with event tracking")
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

    async def process_market_update(self, symbol: str, data: Dict) -> None:
        """Process market data update with news correlation"""
        try:
            # Store price and volume history
            self.price_history[symbol].append({
                'price': float(data['price']),
                'timestamp': datetime.now()
            })
            
            self.volume_history[symbol].append({
                'volume': float(data['volume']),
                'timestamp': datetime.now()
            })
            
            # Detect significant events
            await self._detect_market_events(symbol, data)
            
            # Cleanup old data
            self._cleanup_history(symbol)
            
        except Exception as e:
            logger.error(f"Error processing market update: {e}")

    async def _detect_market_events(self, symbol: str, data: Dict) -> None:
        """Detect significant market events for news correlation"""
        try:
            prices = [p['price'] for p in self.price_history[symbol][-20:]]
            volumes = [v['volume'] for v in self.volume_history[symbol][-20:]]
            
            if len(prices) < 2:
                return
                
            # Calculate metrics
            price_change = (prices[-1] - prices[-2]) / prices[-2]
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            current_volume = volumes[-1]
            
            # Detect events
            if abs(price_change) > self.volatility_threshold:
                event = {
                    'type': 'price_move',
                    'change': price_change,
                    'timestamp': datetime.now()
                }
                self.price_events[symbol].append(event)
                logger.info(f"Significant price move detected for {symbol}: {price_change:.2%}")
                
            if current_volume > avg_volume * self.volume_threshold:
                event = {
                    'type': 'volume_spike',
                    'ratio': current_volume / avg_volume,
                    'timestamp': datetime.now()
                }
                self.price_events[symbol].append(event)
                logger.info(f"Volume spike detected for {symbol}: {current_volume/avg_volume:.2f}x average")
                
        except Exception as e:
            logger.error(f"Error detecting market events: {e}")

    def _cleanup_history(self, symbol: str) -> None:
        """Clean up old historical data"""
        cutoff = datetime.now() - self.correlation_window
        
        self.price_history[symbol] = [
            p for p in self.price_history[symbol]
            if p['timestamp'] > cutoff
        ]
        
        self.volume_history[symbol] = [
            v for v in self.volume_history[symbol]
            if v['timestamp'] > cutoff
        ]
        
        self.price_events[symbol] = [
            e for e in self.price_events[symbol]
            if e['timestamp'] > cutoff
        ]

    async def get_correlated_events(self, symbol: str, 
                                  start_time: datetime) -> List[Dict]:
        """Get market events that may correlate with news"""
        events = []
        
        for event in self.price_events[symbol]:
            if event['timestamp'] >= start_time:
                events.append(event)
                
        return events

class MarketDataFeed(BaseService):
    """Market data feed handler"""
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
        self.market_data_service = None

    async def _setup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.market_data_service = MarketDataService(self.config)
            await self.market_data_service.start()
            self.running = True
            logger.info("Market data feed setup complete")
        except Exception as e:
            logger.error(f"Market data feed setup failed: {e}")
            raise

    async def _cleanup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.running = False
            if self.market_data_service:
                await self.market_data_service.stop()
            self.cache.clear()
            self.data_handlers.clear()
            logger.info("Market data feed cleanup complete")
        except Exception as e:
            logger.error(f"Market data feed cleanup failed: {e}")
            raise

    async def stop(self) -> None:
        """Stop the data feed"""
        await self._cleanup()
        logger.info("Market data feed stopped")

    async def subscribe(self, handler) -> None:
        """Subscribe to market data updates"""
        if handler not in self.data_handlers:
            self.data_handlers.append(handler)
            logger.info(f"Handler {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'} subscribed")

    async def unsubscribe(self, handler) -> None:
        """Unsubscribe from market data updates"""
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)
            logger.info(f"Handler {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'} unsubscribed")

    async def _notify_handlers(self, event_type: str, data: Dict) -> None:
        """Notify all subscribed handlers"""
        for handler in self.data_handlers:
            try:
                await handler(event_type, data)
            except Exception as e:
                logger.error(f"Handler error: {e}")

# Export both classes
__all__ = ['MarketDataService', 'MarketDataFeed']
