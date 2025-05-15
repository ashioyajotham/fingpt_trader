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
        try:
            # Get exchange credentials
            self.api_key = self.config.get('api_key') or os.getenv("BINANCE_API_KEY")
            self.api_secret = self.config.get('api_secret') or os.getenv("BINANCE_API_SECRET")
            
            if not self.api_key or not self.api_secret:
                logger.error("Exchange API credentials not configured")
                raise ValueError("API credentials missing")
            
            # Initialize exchange client
            from services.exchanges.binance import BinanceClient
            self.exchange = await BinanceClient.get_instance({
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'test_mode': self.config.get('test_mode', True)
            })
            
            if not self.exchange:
                raise ValueError("Failed to initialize exchange client")
                
            # Initialize data structures
            self.cache = {}
            self.price_history = {}
            self.volume_history = {}
            
            for pair in self.config.get('pairs', []):
                self.price_history[pair] = []
                self.volume_history[pair] = []
                
            logger.info("Market data service initialized successfully")
        except Exception as e:
            logger.error(f"Market data service initialization failed: {str(e)}")
            raise  # Re-raise to ensure caller knows initialization failed

    async def setup(self, exchange_clients):
        """Set up the market data service with exchange clients"""
        try:
            # Store the Binance client specifically
            if 'binance' in exchange_clients:
                self.exchange_client = exchange_clients['binance']
                logger.info("MarketDataService initialized with Binance client")
            else:
                logger.warning("Binance client not provided to MarketDataService")
                
            # Initialize data containers
            self._market_data = {}
            self.price_history = {}
            
            # Set up initial price caches for configured pairs
            pairs = self.config.get('pairs', [])
            for pair in pairs:
                if pair not in self._market_data:
                    self._market_data[pair] = {
                        'price': None,
                        'timestamp': None,
                        'change': 0.0
                    }
                    self.price_history[pair] = []
                    
            logger.info(f"MarketDataService setup complete for {len(pairs)} pairs")
            return True
        except Exception as e:
            logger.error(f"Error setting up MarketDataService: {e}")
            return False

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.exchange:
            await self.exchange.close()
        self.cache.clear()

    async def get_realtime_quote(self, symbols):
        """Get real-time quotes for multiple symbols with enhanced error handling"""
        quotes = {}
        
        for symbol in symbols:
            try:
                # Check if exchange_client exists
                if not hasattr(self, 'exchange_client') or self.exchange_client is None:
                    logger.error(f"No exchange client available for {symbol}")
                    continue
                    
                # Get quote from exchange
                quote = await self.exchange_client.get_ticker(symbol)
                
                if quote:
                    price = None
                    # Try different field names that might contain price
                    for field in ['lastPrice', 'price', 'last', 'close']:
                        if field in quote and quote[field]:
                            try:
                                price = float(quote[field])
                                if price > 0:
                                    break
                            except (ValueError, TypeError):
                                pass
                    
                    if price and price > 0:
                        # Store in market data cache
                        if not hasattr(self, '_market_data'):
                            self._market_data = {}
                        self._market_data[symbol] = {
                            'price': price,
                            'timestamp': datetime.now()
                        }
                        quotes[symbol] = price
                        
                        # Update price history for 24h change calculation
                        self.update_price_history(symbol, price)
                        
                        # Log success
                        logger.info(f"Fetched price for {symbol}: {price}")
                    else:
                        logger.warning(f"Invalid price data for {symbol}: {quote}")
                else:
                    logger.warning(f"Empty quote data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
                # Print detailed exception for debugging
                import traceback
                logger.error(traceback.format_exc())
        
        return quotes

    async def fetch_prices_directly(self, symbols):
        """Fetch prices directly from Binance REST API as a fallback"""
        quotes = {}
        try:
            # Use context manager for automatic connection cleanup
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for symbol in symbols:
                    try:
                        # Use Binance public API with retries and rate limiting
                        for attempt in range(3):  # 3 retries
                            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                            try:
                                async with session.get(url) as response:
                                    if response.status == 200:
                                        data = await response.json()
                                        if 'price' in data:
                                            price = float(data['price'])
                                            if price > 0:
                                                logger.info(f"Fetched price directly for {symbol}: {price}")
                                                quotes[symbol] = price
                                                
                                                # Update internal tracking
                                                if not hasattr(self, '_market_data'):
                                                    self._market_data = {}
                                                self._market_data[symbol] = {
                                                    'price': price,
                                                    'timestamp': datetime.now(),
                                                    'source': 'direct_api'
                                                }
                                                self.update_price_history(symbol, price)
                                                # Successful fetch, break retry loop
                                                break
                                    elif response.status == 429:
                                        # Rate limit hit - exponential backoff
                                        backoff = (2 ** attempt) * 0.1
                                        logger.warning(f"Rate limit hit, backing off for {backoff}s")
                                        await asyncio.sleep(backoff)
                                        continue
                                    else:
                                        logger.warning(f"Failed to fetch price for {symbol}: HTTP {response.status}")
                                        if attempt < 2:  # Don't sleep on last attempt
                                            await asyncio.sleep(0.5)
                            except aiohttp.ClientError as ce:
                                logger.error(f"Connection error for {symbol}: {ce}")
                                if attempt < 2:
                                    await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error fetching direct price for {symbol}: {e}")
                
                return quotes
        except Exception as e:
            logger.error(f"Critical error in direct price fetching: {e}")
            return {}

    async def _fetch_with_retry(self, symbol: str) -> Dict:
        """Fetch market data with retry logic"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Try get_ticker instead of get_symbol_ticker
                data = await self.exchange.get_ticker(symbol=symbol)
                logger.debug(f"Raw ticker data structure: {data}")
                
                # Extract price correctly based on response format
                if isinstance(data, dict) and 'price' not in data and 'lastPrice' in data:
                    # Fix for Binance response format
                    data['price'] = data['lastPrice']
                
                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {str(e)}")
                await asyncio.sleep(1)
        
        logger.error(f"Error fetching {symbol} after {max_retries} attempts")
        return {
            "symbol": symbol,
            "price": None,
            "error": "Failed after multiple attempts",
            "timestamp": datetime.now().timestamp()
        }

    async def _try_fallback_exchanges(self, symbol: str) -> dict:
        """Attempt to get data from fallback exchanges when primary fails"""
        logger.info(f"Using fallback exchanges for {symbol}")
        
        # Try alternative exchanges defined in config
        fallback_exchanges = self.config.get('fallback_exchanges', [])
        
        # Fix: Check if we have a single exchange client instead of a dictionary
        if hasattr(self, 'exchange'):
            try:
                # Try with the main exchange again with different method
                data = await self.exchange.get_symbol_ticker(symbol=symbol)
                logger.info(f"Successfully fetched {symbol} data using alternative method")
                return data
            except Exception as e:
                logger.warning(f"Alternative method also failed: {e}")
        
        # Rest of the fallback logic...

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

    def get_watched_pairs(self):
        """
        Get list of trading pairs being monitored.
        
        Returns:
            List[str]: List of trading pair symbols
        """
        return self.config.get('pairs', [])

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol"""
        try:
            # First check portfolio last_prices as it's more reliable
            if hasattr(self, 'last_prices') and symbol in self.last_prices:
                return float(self.last_prices[symbol])
                
            # Then try market_data
            if hasattr(self, '_market_data') and symbol in self._market_data:
                if 'price' in self._market_data[symbol]:
                    return float(self._market_data[symbol]['price'])
                    
            # Check alternate structures
            if hasattr(self, 'market_data') and symbol in self.market_data:
                if 'price' in self.market_data[symbol]:
                    return float(self.market_data[symbol]['price'])
                    
            # Check cache
            if hasattr(self, 'cache') and symbol in self.cache:
                for field in ['price', 'lastPrice', 'last', 'close']:
                    if field in self.cache[symbol]:
                        try:
                            return float(self.cache[symbol][field])
                        except (ValueError, TypeError):
                            pass
                            
            # Log the issue instead of using fallbacks
            logger.warning(f"No price data available for {symbol}")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_latest_data(self, symbol: str) -> Dict:
        """Return complete market data for UI display"""
        result = {
            'price': 0.0,
            'change': 0.0,
            'volume': 0.0,
            'high': 0.0,
            'low': 0.0,
            'sentiment': 'Neutral'
        }
        
        # If we have cached market data, use it
        if hasattr(self, 'market_data') and symbol in self.market_data:
            data = self.market_data[symbol]
            
            # Extract price from various possible fields
            for field in ['price', 'lastPrice', 'last', 'close']:
                if field in data and data[field]:
                    try:
                        result['price'] = float(data[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Extract other fields if available
            if 'change' in data:
                try:
                    result['change'] = float(data['change'])
                except (ValueError, TypeError):
                    pass
                    
            if 'volume' in data:
                try:
                    result['volume'] = float(data['volume'])
                except (ValueError, TypeError):
                    pass
                    
            if 'high' in data:
                try:
                    result['high'] = float(data['high'])
                except (ValueError, TypeError):
                    pass
                    
            if 'low' in data:
                try:
                    result['low'] = float(data['low'])
                except (ValueError, TypeError):
                    pass
        
        return result

    def update_symbol_data(self, symbol: str, data: Dict) -> None:
        """Update symbol-specific data in the market data store"""
        if not hasattr(self, 'market_data'):
            self.market_data = {}
        
        # If symbol isn't in market_data yet, initialize it
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        # Update the data
        self.market_data[symbol].update(data)
        
        # Update last updated timestamp
        self.market_data[symbol]['last_updated'] = datetime.now()
        
        # Log the update for debugging
        logger.debug(f"Updated market data for {symbol}: {data}")

    def _sync_market_data_from_cache(self):
        """Ensure market_data has all the data from cache for consistency"""
        if not hasattr(self, 'cache'):
            self.cache = {}
            
        if not hasattr(self, 'market_data'):
            self.market_data = {}
            
        # Transfer all cache data to market_data
        for symbol, data in self.cache.items():
            if isinstance(data, dict):
                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                self.market_data[symbol].update(data)

    def check_data_quality(self):
        """Check the quality of available market data"""
        if not hasattr(self, 'market_data'):
            logger.error("No market data available")
            return False
            
        missing_prices = []
        zero_prices = []
        
        for symbol, data in self.market_data.items():
            price = None
            for field in ['price', 'lastPrice', 'last', 'close']:
                if field in data and data[field]:
                    try:
                        price = float(data[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            if price is None:
                missing_prices.append(symbol)
            elif price <= 0:
                zero_prices.append(symbol)
        
        if missing_prices:
            logger.warning(f"Missing price data for: {', '.join(missing_prices)}")
            
        if zero_prices:
            logger.warning(f"Zero or invalid prices for: {', '.join(zero_prices)}")
        
        return not (missing_prices or zero_prices)

    def calculate_price_changes(self) -> Dict[str, float]:
        """Calculate 24h price changes with optimized cache usage"""
        changes = {}
        
        # First use the change cache for instant results
        if hasattr(self, 'change_cache'):
            for symbol, data in self.change_cache.items():
                changes[symbol] = data['current_change']
        
        # For any missing symbols, use the existing calculation method
        for symbol, data in self.price_history.items():
            if symbol not in changes:
                if data:
                    newest = data[-1]['price'] if data else 0
                    oldest = data[0]['price'] if data else 0
                    if oldest and oldest > 0:
                        change_pct = ((newest - oldest) / oldest) * 100
                        changes[symbol] = round(change_pct, 2)
                    else:
                        changes[symbol] = 0.0
                else:
                    changes[symbol] = 0.0
        
        return changes

    def update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol with optimized change tracking"""
        if not hasattr(self, 'price_history'):
            self.price_history = {}
        
        if not hasattr(self, 'change_cache'):
            self.change_cache = {}
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        if symbol not in self.change_cache:
            self.change_cache[symbol] = {
                'reference_price': price,
                'reference_time': datetime.now(),
                'current_change': 0.0
            }
        
        current_time = datetime.now()
        
        # Add new price data point with timestamp
        self.price_history[symbol].append({
            'price': price,
            'timestamp': current_time
        })
        
        # Limit history size (keep 24h worth of data)
        max_history = 24 * 60  # 24 hours of minute data
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Update the market_data store
        if not hasattr(self, '_market_data'):
            self._market_data = {}
        
        if symbol not in self._market_data:
            self._market_data[symbol] = {}
        
        self._market_data[symbol]['price'] = price
        self._market_data[symbol]['timestamp'] = current_time
        
        # Update 24h change cache - key optimization
        ref_entry = self.change_cache[symbol]
        ref_time = ref_entry['reference_time']
        ref_price = ref_entry['reference_price']
        time_diff = (current_time - ref_time).total_seconds() / 3600  # hours
        
        # If reference point is close to 24h old, update the change
        if 23 <= time_diff <= 25:
            # We have a good 24h reference point
            if ref_price > 0:
                percent_change = ((price - ref_price) / ref_price) * 100
                ref_entry['current_change'] = round(percent_change, 2)
            
            # Reset reference point to current for next 24h
            ref_entry['reference_price'] = price
            ref_entry['reference_time'] = current_time
        elif time_diff > 25:
            # Reference too old, reset
            ref_entry['reference_price'] = price
            ref_entry['reference_time'] = current_time
            ref_entry['current_change'] = 0.0
        
        # Update the change in market data
        self._market_data[symbol]['change'] = ref_entry['current_change']

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

    async def update_prices(self, pairs: List[str]) -> None:
        """Update prices for the given trading pairs."""
        current_prices = {}
        # Add defensive coding
        for pair in pairs:
            try:
                # Get latest price with additional logging
                if hasattr(self.market_data_service, 'get_latest_price'):
                    price = self.market_data_service.get_latest_price(pair)
                    if price is None:
                        logger.warning(f"get_latest_price for {pair} returned None")
                    elif price <= 0:
                        logger.warning(f"get_latest_price for {pair} returned invalid price: {price}")
                    else:
                        logger.debug(f"Got valid price for {pair}: {price}")
                    current_prices[pair] = price
                else:
                    logger.warning("Market data service missing get_latest_price method")
            except Exception as e:
                logger.error(f"Error getting price for {pair}: {str(e)}")

        market_data = await self.market_data_service.get_realtime_quote(pairs)
        logger.debug(f"Raw market data: {market_data}")

        # Update the cache directly for immediate fix
        for symbol, data in market_data.items():
            if data and isinstance(data, dict):
                self.market_data_service.cache[symbol] = data
                
                # Try to extract price using several common field names
                price = None
                for field in ['price', 'lastPrice', 'last', 'close']:
                    if field in data and data[field]:
                        try:
                            price = float(data[field])
                            if price > 0:
                                logger.info(f"Found valid price for {symbol}: {price} (field: {field})")
                                break
                        except (ValueError, TypeError):
                            pass
                            
                # If we found a valid price, store it directly in price_history
                if price and price > 0:
                    if symbol not in self.market_data_service.price_history:
                        self.market_data_service.price_history[symbol] = []
                    self.market_data_service.price_history[symbol].append({
                        'price': price,
                        'timestamp': datetime.now()
                    })

# Export both classes
__all__ = ['MarketDataService', 'MarketDataFeed']
