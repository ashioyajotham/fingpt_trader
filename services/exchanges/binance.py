import logging
from typing import Dict, List, Optional
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from .base import BaseExchangeClient
import asyncio
import sys
import platform
from aiohttp import TCPConnector, ClientSession

logger = logging.getLogger(__name__)

class BinanceClient(BaseExchangeClient):
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
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
        """Initialize Binance client with Windows compatibility"""
        try:
            # Windows-specific connector settings
            connector_kwargs = {}
            if platform.system() == 'Windows':
                connector_kwargs['family'] = 4  # IPv4 only
                
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                session_kwargs={
                    'connector': TCPConnector(
                        ssl=True,
                        **connector_kwargs
                    )
                }
            )
            
            self.bsm = BinanceSocketManager(self.client)
            logger.info("Binance client initialized")
        except Exception as e:
            logger.error(f"Binance client initialization failed: {e}")
            raise

    async def cleanup(self):
        """Enhanced cleanup with websocket handling"""
        try:
            # Close websocket connections
            for symbol, streams in self._ws_connections.items():
                for stream in streams.values():
                    await stream.close()
            self._ws_connections.clear()
            
            # Close main client
            if self.client:
                await self.client.close_connection()
                self.client = None
                
            logger.info("Binance client cleaned up")
        except Exception as e:
            logger.error(f"Binance cleanup failed: {e}")

    async def get_trading_pairs(self) -> List[str]:
        """Get available trading pairs"""
        try:
            exchange_info = await self.client.get_exchange_info()
            return [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            raise

    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for symbol"""
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
        """Get recent trades for symbol"""
        try:
            trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            raise

    async def get_candles(self, symbol: str, interval: str = '1h', limit: int = 500) -> List[List]:
        """Get candlestick data"""
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
        """Implement abstract method for market data"""
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
        """Implement abstract method for order placement"""
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