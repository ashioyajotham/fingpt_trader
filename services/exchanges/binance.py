"""
Binance Exchange Client

Provides a robust interface to Binance API with:
- Automatic connection management
- Request retry logic
- Rate limit handling
- Resource cleanup
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)

class BinanceClient:
    """Async Binance exchange interface"""
    
    ENDPOINTS = {
        'test': {
            'rest': 'https://testnet.binance.vision/api',
            'ws': 'wss://testnet.binance.vision/ws'
        },
        'prod': {
            'rest': 'https://api.binance.com/api',
            'ws': 'wss://stream.binance.com:9443/ws'
        }
    }

    def __init__(self, api_key: str, api_secret: str):
        """Initialize with API credentials"""
        if not api_key or not api_secret:
            raise ValueError("API credentials required")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.session = None
        self.bsm = None
        self._ws_connections = {}
        self._initialized = False
        self.testnet = False
        self.streams = {}
        self.data_handlers = {}
        self.orderbook_manager = {}
        self.trade_stream_handlers = {}
        self.kline_stream_handlers = {}

    @classmethod
    async def create(cls, config: Dict) -> 'BinanceClient':
        """Factory method for client creation"""
        if not all(k in config for k in ('api_key', 'api_secret')):
            raise ValueError("Missing required credentials")
            
        instance = cls(
            api_key=config['api_key'],
            api_secret=config['api_secret']
        )
        
        instance.testnet = config.get('test_mode', True)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        """Initialize client connection"""
        if self._initialized:
            return
            
        try:
            # Create main client
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Create session for custom requests
            self.session = aiohttp.ClientSession()
            
            # Test connection
            await self.client.ping()
            
            self._initialized = True
            logger.info(f"Initialized Binance client (testnet: {self.testnet})")
            
        except Exception as e:
            await self.cleanup()
            raise ValueError(f"Failed to initialize Binance client: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources"""
        # Close websocket connections
        for symbol, streams in self._ws_connections.items():
            for stream in streams.values():
                try:
                    await stream.close()
                except:
                    pass
        self._ws_connections.clear()
        
        # Close session
        if self.session:
            await self.session.close()
            
        # Close main client
        if self.client:
            await self.client.close_connection()
            
        self._initialized = False
        logger.info("Cleaned up Binance resources")

    async def get_trading_pairs(self) -> List[str]:
        """Get active trading pairs"""
        info = await self.client.get_exchange_info()
        return [s['symbol'] for s in info['symbols'] 
                if s['status'] == 'TRADING']

    async def start_market_streams(self, symbols: List[str]) -> None:
        """Start real-time market data streams"""
        if not self.bsm:
            self.bsm = BinanceSocketManager(self.client)
            
        for symbol in symbols:
            # Start order book stream
            self.streams[f"{symbol}_book"] = await self.bsm.depth_socket(
                symbol, depth=BinanceSocketManager.WEBSOCKET_DEPTH_20
            )
            
            # Start trade stream
            self.streams[f"{symbol}_trade"] = await self.bsm.trade_socket(symbol)
            
            # Start kline stream
            self.streams[f"{symbol}_kline"] = await self.bsm.kline_socket(
                symbol, interval='1m'
            )
            
            # Initialize order book
            self.orderbook_manager[symbol] = {
                'bids': {},
                'asks': {},
                'last_update': None
            }
            
            logger.info(f"Started market streams for {symbol}")

    async def _handle_orderbook(self, msg: Dict) -> None:
        """Process orderbook updates"""
        symbol = msg['s']
        
        if 'b' in msg:  # Bids update
            for bid in msg['b']:
                price, qty = Decimal(bid[0]), Decimal(bid[1])
                if qty == 0:
                    self.orderbook_manager[symbol]['bids'].pop(price, None)
                else:
                    self.orderbook_manager[symbol]['bids'][price] = qty
                    
        if 'a' in msg:  # Asks update
            for ask in msg['a']:
                price, qty = Decimal(ask[0]), Decimal(ask[1])
                if qty == 0:
                    self.orderbook_manager[symbol]['asks'].pop(price, None)
                else:
                    self.orderbook_manager[symbol]['asks'][price] = qty
                    
        self.orderbook_manager[symbol]['last_update'] = datetime.now()

    def get_orderbook_snapshot(self, symbol: str) -> Dict:
        """Get current orderbook snapshot"""
        ob = self.orderbook_manager.get(symbol, {})
        return {
            'bids': sorted(
                [(float(k), float(v)) for k, v in ob.get('bids', {}).items()],
                reverse=True
            )[:20],
            'asks': sorted(
                [(float(k), float(v)) for k, v in ob.get('asks', {}).items()],
            )[:20],
            'timestamp': ob.get('last_update')
        }

    async def process_trade(self, msg: Dict) -> Dict:
        """Normalize trade data"""
        return {
            'symbol': msg['s'],
            'price': float(msg['p']),
            'quantity': float(msg['q']),
            'time': pd.Timestamp(msg['T'], unit='ms'),
            'is_buyer_maker': msg['m'],
            'trade_id': msg['t']
        }

    async def process_kline(self, msg: Dict) -> Dict:
        """Normalize kline/candlestick data"""
        k = msg['k']
        return {
            'symbol': msg['s'],
            'interval': k['i'],
            'time': pd.Timestamp(k['t'], unit='ms'),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v']),
            'closed': k['x']
        }

    async def start_data_processing(self) -> None:
        """Start processing market data streams"""
        try:
            tasks = []
            for stream_id, stream in self.streams.items():
                symbol = stream_id.split('_')[0]
                stream_type = stream_id.split('_')[1]
                
                if stream_type == 'book':
                    tasks.append(self._process_stream(
                        stream, self._handle_orderbook
                    ))
                elif stream_type == 'trade':
                    tasks.append(self._process_stream(
                        stream, self.process_trade
                    ))
                elif stream_type == 'kline':
                    tasks.append(self._process_stream(
                        stream, self.process_kline
                    ))
                    
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise

    async def _process_stream(self, stream, handler) -> None:
        """Process individual data stream"""
        async with stream as tscm:
            while True:
                try:
                    msg = await tscm.recv()
                    await handler(msg)
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    await asyncio.sleep(1)