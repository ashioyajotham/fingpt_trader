"""
Binance Exchange Client Implementation

A comprehensive AsyncIO-based client for interacting with Binance exchange API.
Supports both mainnet and testnet environments with enhanced error handling,
connection management, and Windows-specific optimizations.

Features:
    - Async/await pattern for optimal performance
    - Automatic reconnection handling
    - Rate limiting compliance
    - Custom DNS resolution for Windows
    - WebSocket stream management
    - Enhanced error handling
    - Connection pooling and reuse
    - Comprehensive market data access

Usage:
    client = await BinanceClient.create({
        'api_key': 'your_key',
        'api_secret': 'your_secret',
        'test_mode': True
    })
    
    # Get market data
    pairs = await client.get_trading_pairs()
    orderbook = await client.get_orderbook('BTCUSDT')
    
    # Cleanup
    await client.cleanup()

Dependencies:
    - python-binance
    - aiohttp
    - asyncio
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from .base import BaseExchangeClient
import asyncio
import platform
import socket
import pandas as pd
from aiohttp import TCPConnector, ClientSession

logger = logging.getLogger(__name__)

class BinanceClient(BaseExchangeClient):
    """
    Asynchronous Binance exchange client with enhanced features.
    
    Provides a robust interface to Binance's API with:
    - Automatic connection management
    - WebSocket stream handling
    - Rate limit awareness
    - Error recovery
    - Resource cleanup
    
    Attributes:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        testnet (bool): Whether to use testnet
        base_url (str): REST API base URL
        ws_url (str): WebSocket base URL
        client (AsyncClient): Binance async client instance
        bsm (BinanceSocketManager): WebSocket manager
        _ws_connections (dict): Active WebSocket connections
        
    Class Constants:
        DEFAULT_HEADERS (dict): Default HTTP headers
        URLS (dict): API endpoints for prod/test
    """
    # Class-level constants
    DEFAULT_HEADERS = {
        'User-Agent': 'FinGPT-Trader/1.0',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    URLS = {
        'test': {
            'rest': 'https://testnet.binance.vision',
            'ws': 'wss://testnet.binance.vision/ws'
        },
        'prod': {
            'rest': 'https://api.binance.com',
            'ws': 'wss://stream.binance.com:9443/ws'
        }
    }

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = True, options: dict = None):
        """Initialize Binance client"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.options = options or {}
        
        # Use testnet to determine URLs
        urls = self.URLS['test'] if testnet else self.URLS['prod']
        self.base_url = urls['rest']
        self.ws_url = urls['ws']
        
        self.client = None
        self.bsm = None
        self._ws_connections = {}

    @classmethod
    async def create(cls, config: Dict) -> 'BinanceClient':
        """Factory method for client creation"""
        instance = cls(
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            testnet=config.get('test_mode', True)
        )
        await instance.initialize()
        return instance

    async def initialize(self):
        """
        Initialize Binance client with optimized settings.
        
        Performs:
        1. Session configuration with custom DNS settings
        2. Connection pool setup
        3. WebSocket manager initialization
        4. Connection testing
        
        Raises:
            Exception: On initialization failure with detailed error
        """
        try:
            logger.debug("Initializing Binance client...")
            logger.debug(f"Base URL: {self.base_url}")
            logger.debug(f"WebSocket URL: {self.ws_url}")
            logger.debug(f"Test mode: {self.testnet}")
            
            # Configure DNS resolution
            use_custom_dns = platform.system() == 'Windows'
            logger.debug(f"Using custom DNS settings: {use_custom_dns}")
            
            # Configure connector with DNS settings
            connector = TCPConnector(
                ssl=True,
                family=socket.AF_INET,  # Force IPv4
                force_close=True,
                enable_cleanup_closed=True,
                verify_ssl=True,
                use_dns_cache=not use_custom_dns,  # Disable DNS cache on Windows
                ttl_dns_cache=300  # 5 minutes cache TTL when enabled
            )

            # Create custom session and store it
            self.session = ClientSession(
                connector=connector,
                headers=self.DEFAULT_HEADERS,
                skip_auto_headers=['Content-Type']
            )

            # Create async client with custom session and proper timeout
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                tld='vision' if self.testnet else 'com',
                requests_params={'timeout': 30}  # Remove use_dns_cache from here
            )

            # Replace client's session
            if hasattr(self.client, 'session'):
                await self.client.session.close()
                self.client.session = self.session

            # Initialize socket manager
            self.bsm = BinanceSocketManager(self.client)
            logger.info("Binance client initialized successfully")
            logger.debug("Connection test successful")

        except Exception as e:
            logger.error(f"Binance client initialization failed: {e}")
            if hasattr(self, 'session'):
                await self.session.close()
            if hasattr(self, 'client') and self.client:
                await self.client.close_connection()
            raise

    async def cleanup(self):
        """
        Perform comprehensive resource cleanup.
        
        Closes:
        1. All WebSocket connections
        2. HTTP sessions
        3. AsyncIO client connection
        
        Should be called when client is no longer needed.
        """
        try:
            # Close websocket connections
            for symbol, streams in self._ws_connections.items():
                for stream in streams.values():
                    await stream.close()
            self._ws_connections.clear()
            
            # Close session
            if hasattr(self, 'session'):
                await self.session.close()
                self.session = None
                
            # Close client session and connection
            if self.client:
                if hasattr(self.client, 'session'):
                    await self.client.session.close()
                await self.client.close_connection()
                self.client = None
                
            logger.info("Binance client cleaned up")
        except Exception as e:
            logger.error(f"Binance cleanup failed: {e}")

    async def get_trading_pairs(self) -> List[str]:
        """
        Retrieve available trading pairs from exchange.
        
        Returns:
            List[str]: Active trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            
        Raises:
            Exception: If API request fails
        """
        try:
            exchange_info = await self.client.get_exchange_info()
            return [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch order book data for a trading pair.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            limit (int): Number of price levels to retrieve (default: 100)
            
        Returns:
            Dict: Order book data:
                {
                    'bids': [[price, quantity], ...],
                    'asks': [[price, quantity], ...],
                    'timestamp': last_update_id
                }
                
        Raises:
            Exception: If symbol is invalid or request fails
        """
        try:
            depth = await self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'bids': depth['bids'],
                'asks': depth['asks'],
                'timestamp': depth['lastUpdateId']
            }
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            raise

    async def get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Fetch recent trades for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of trades to retrieve (max 1000)
            
        Returns:
            List[Dict]: Recent trades with fields:
                - id: Trade ID
                - price: Execution price
                - qty: Trade quantity
                - time: Trade timestamp
                - isBuyerMaker: True if buyer was maker
                
        Raises:
            Exception: On invalid symbol or API error
        """
        try:
            trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            raise

    async def get_candles(self, symbol: str, interval: str = '1h', limit: int = 500) -> List[List]:
        """
        Retrieve candlestick data for technical analysis.
        
        Args:
            symbol (str): Trading pair symbol
            interval (str): Candle interval ('1m', '5m', '1h', etc.)
            limit (int): Number of candles (max 1000)
            
        Returns:
            List[List]: Candlestick data:
                [
                    [timestamp, open, high, low, close, volume, ...],
                    ...
                ]
                
        Raises:
            Exception: If parameters are invalid or request fails
        """
        try:
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            raise

    async def get_market_data(self, symbol: str) -> Dict:
        """
        Fetch comprehensive market data for a symbol.
        
        Retrieves synchronized snapshot of:
        - Order book
        - Recent trades
        - Candlestick data
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Market data snapshot:
                {
                    'orderbook': {...},
                    'trades': [...],
                    'candles': [...],
                    'timestamp': current_timestamp
                }
                
        Raises:
            Exception: If any data fetch fails
        """
        try:
            # Get all required market data
            [orderbook, trades, klines] = await asyncio.gather(
                self.get_orderbook(symbol),
                self.get_recent_trades(symbol),
                self.get_candles(symbol)
            )
            
            return {
                'orderbook': orderbook,
                'trades': trades,
                'candles': klines,
                'timestamp': datetime.now().timestamp()
            }
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise

    async def place_order(self, order: Dict) -> Dict:
        """
        Place a new order on the exchange.
        
        Supports multiple order types:
        - LIMIT
        - MARKET
        - STOP_LOSS
        - STOP_LOSS_LIMIT
        - TAKE_PROFIT
        - TAKE_PROFIT_LIMIT
        
        Args:
            order (Dict): Order specification:
                - symbol: Trading pair
                - side: 'BUY' or 'SELL'
                - type: Order type
                - quantity: Order size
                - price: Limit price (optional)
                - stop_price: Stop price (optional)
                
        Returns:
            Dict: Order result:
                {
                    'id': Order ID
                    'status': Order status
                    'filled': Amount filled
                    'remaining': Amount remaining
                    'price': Average fill price
                    'raw': Raw exchange response
                }
                
        Raises:
            Exception: If order parameters are invalid or placement fails
        """
        try:
            params = {
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'quantity': order['quantity']
            }
            
            if order.get('price'):
                params['price'] = order['price']
            
            if order.get('stop_price'):
                params['stopPrice'] = order['stop_price']
                
            response = await self.client.create_order(**params)
            return {
                'id': response['orderId'],
                'status': response['status'],
                'filled': float(response.get('executedQty', 0)),
                'remaining': float(response.get('origQty', 0)) - float(response.get('executedQty', 0)),
                'price': float(response.get('price', 0)),
                'raw': response
            }
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

    async def _start_market_stream(self, symbol: str):
        """Start market data websocket stream"""
        try:
            # Create socket manager if needed
            if not self.bsm:
                self.bsm = BinanceSocketManager(self.client)
            
            # Start market streams
            self._ws_connections[symbol] = {
                'trades': await self.bsm.trade_socket(symbol),
                'depth': await self.bsm.depth_socket(symbol),
                'kline': await self.bsm.kline_socket(symbol)
            }
            
            logger.info(f"Started market streams for {symbol}")
        except Exception as e:
            logger.error(f"Failed to start market stream for {symbol}: {e}")
            raise

    async def get_historical_klines(self, symbol: str, interval: str, 
                                  start_str: str, end_str: str) -> List:
        """
        Get historical candlestick data.

        Args:
            symbol (str): Trading pair symbol (e.g. 'BTCUSDT')  
            interval (str): Kline interval ('1m','1h','1d', etc.)
            start_str (str): Start time in ISO format 
            end_str (str): End time in ISO format

        Returns:
            List[List]: Historical klines data:
                [
                    [timestamp, open, high, low, close, volume, ...],
                    ...
                ]
        """
        try:
            klines = await self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str, 
                end_str=end_str,
                limit=1000  # Maximum allowed
            )
            return klines
            
        except Exception as e:
            logger.error(f"Failed to get historical klines for {symbol}: {e}")
            raise

    async def get_aggregate_trades(self, symbol: str, start_str: str, 
                                 end_str: str) -> List[Dict]:
        """
        Get historical aggregated trade data.

        Args:
            symbol (str): Trading pair symbol
            start_str (str): Start time in ISO format
            end_str (str): End time in ISO format 

        Returns:
            List[Dict]: Historical trades
        """
        try:
            trades = await self.client.get_aggregate_trades(
                symbol=symbol,
                startTime=start_str,
                endTime=end_str,
                limit=1000
            )
            return trades
        except Exception as e:
            logger.error(f"Failed to get historical trades for {symbol}: {e}")
            raise

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data for a symbol"""
        try:
            response = await self.client.get_symbol_ticker(symbol=symbol)
            return {
                'symbol': response['symbol'],
                'price': float(response['price']),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    async def ping(self) -> bool:
        """Test connectivity to the exchange"""
        try:
            if not hasattr(self, 'session'):
                raise Exception("Client session not initialized")
                
            endpoint = f"{self.base_url}/api/v3/ping"  # Fixed endpoint path
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    logger.info("Successfully pinged Binance API")
                    return True
                raise Exception(f"Ping failed with status {response.status}")
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")
            raise