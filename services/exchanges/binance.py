import logging
from typing import Dict, List, Optional
from datetime import datetime
from .base import BaseExchangeClient
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class BinanceClient(BaseExchangeClient):
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.bsm = None
        
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
        """Initialize Binance client"""
        try:
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.bsm = BinanceSocketManager(self.client)
            logger.info("Binance client initialized")
        except Exception as e:
            logger.error(f"Binance client initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
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