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
from datetime import datetime
import asyncio
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
from binance import AsyncClient, BinanceSocketManager

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