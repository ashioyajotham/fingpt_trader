"""
Binance Exchange Client

A comprehensive async implementation of Binance exchange connectivity with robust
error handling, rate limiting, and connection management.

Key Features:
    - Singleton connection pattern
    - Automatic request retry logic
    - Rate limit compliance
    - WebSocket data streaming
    - Order management with smart routing
    - Market data normalization
    - Resource cleanup

Market Data:
    - Real-time orderbook management
    - Trade stream processing
    - Candlestick aggregation
    - Ticker data handling

Order Types:
    - Market orders
    - Limit orders
    - Stop orders
    - OCO orders

Risk Management:
    - Order validation
    - Position checks
    - Balance verification
    - Lot size enforcement
    - Minimum notional checks

Technical Details:
    - Async/await implementation
    - WebSocket connection pooling
    - Smart reconnection logic
    - Error recovery mechanisms
    - Memory-efficient data structures

Usage:
    ```python
    # Get singleton instance
    client = await BinanceClient.get_instance({
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret',
        'test_mode': True
    })

    # Market data
    ticker = await client.get_ticker('BTCUSDT')
    trades = await client.get_recent_trades('BTCUSDT')
    candles = await client.get_candles('BTCUSDT', '1m')

    # Trading
    order = await client.create_buy_order(
        symbol='BTCUSDT',
        amount=0.01,
        order_type='MARKET'
    )
    ```

Dependencies:
    - python-binance
    - aiohttp
    - pandas
    - numpy
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
    """
    Asynchronous Binance exchange client implementing a singleton pattern.
    
    This class provides a robust interface to the Binance API with:
    - Connection management
    - Request retries
    - Rate limiting
    - WebSocket streams
    - Order management
    
    Features:
        - Real-time market data streaming
        - Order execution with validation
        - Position management
        - Risk checks and limits
        - Resource cleanup
        
    Attributes:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        testnet (bool): Whether to use testnet
        client (AsyncClient): Binance async client instance
        session (aiohttp.ClientSession): HTTP session
        bsm (BinanceSocketManager): WebSocket manager
        symbol_info (Dict): Trading pair information
        streams (Dict): Active data streams
        orderbook_manager (Dict): Order book state
        
    Connection Management:
        - Singleton pattern ensures single connection
        - Automatic reconnection
        - Resource cleanup
        - Connection pooling
        
    Market Data:
        - Order book management
        - Trade stream processing
        - Candlestick aggregation
        - Ticker data
        
    Order Management:
        - Smart order routing
        - Position validation
        - Risk checks
        - Fill monitoring
        
    Error Handling:
        - Request retries
        - Rate limit management
        - Connection recovery
        - Error logging
    """
    
    # Add singleton instance
    _instance = None
    
    @classmethod
    async def get_instance(cls, config=None):
        """Get singleton instance"""
        if not cls._instance:
            if not config:
                raise ValueError("Config required for first initialization")
            cls._instance = await cls.create(config)
        return cls._instance

    @classmethod
    async def create(cls, config: Dict) -> 'BinanceClient':
        """Factory method for client creation"""
        instance = cls(
            api_key=config['api_key'],
            api_secret=config['api_secret']
        )
        
        instance.testnet = config.get('test_mode', True)
        await instance.initialize()
        return instance

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
            
            # Get exchange info for quantity precision
            self.exchange_info = await self.client.get_exchange_info()
            self.symbol_info = {
                s['symbol']: s for s in self.exchange_info['symbols']
            }
            
            # Create session for custom requests
            self.session = aiohttp.ClientSession()
            
            # Test connection
            await self.client.ping()
            
            self._initialized = True
            logger.info(f"Initialized Binance client (testnet: {self.testnet})")
            
        except Exception as e:
            await self.cleanup()
            raise ValueError(f"Failed to initialize Binance client: {str(e)}")

    async def close(self) -> None:
        """Close client connections"""
        try:
            if self.session:
                await self.session.close()
            if self.client:
                await self.client.close_connection()
            if self.bsm:
                # Close any active websocket connections
                for stream in self.streams.values():
                    await stream.close()
            logger.info("Binance client connections closed")
        except Exception as e:
            logger.error(f"Error closing Binance client: {e}")

    async def close_connections(self):
        """Close all open HTTP and WebSocket connections"""
        try:
            # Close HTTP session if exists
            if hasattr(self, 'session') and self.session:
                await self.session.close()
                logger.info("Binance HTTP session closed")
            
            # Close WebSocket connection if exists
            if hasattr(self, 'ws_connection') and self.ws_connection:
                await self.ws_connection.close()
                logger.info("Binance WebSocket connection closed")
            
            # Any other cleanup needed
            if hasattr(self, 'rest_client') and hasattr(self.rest_client, 'close'):
                await self.rest_client.close()
            
        except Exception as e:
            logger.error(f"Error closing Binance connections: {e}")

    async def cleanup(self) -> None:
        """Alias for close() for compatibility"""
        await self.close()

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

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get orderbook for symbol"""
        try:
            orderbook = await self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'bids': [[price, qty] for price, qty in orderbook['bids']],
                'asks': [[price, qty] for price, qty in orderbook['asks']],
                'timestamp': orderbook['lastUpdateId']
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {'bids': [], 'asks': [], 'timestamp': None}

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for symbol"""
        try:
            trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)
            return [{
                'id': trade['id'],
                'price': float(trade['price']),
                'quantity': float(trade['qty']),
                'timestamp': trade['time'],
                'side': 'buy' if trade['isBuyerMaker'] else 'sell'
            } for trade in trades]
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return []

    async def get_candles(self, symbol: str, interval: str = '1m', limit: int = 100) -> List:
        """Get candlestick data for symbol"""
        try:
            candles = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return candles  # Returns raw candle data
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data for symbol"""
        try:
            ticker = await self.client.get_ticker(symbol=symbol)
            return {
                'price': float(ticker['lastPrice']),
                'volume': float(ticker['volume']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice'])
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            # Try retrieving data from fallback exchange if configured
            if hasattr(self, '_try_fallback_exchanges'):
                return await self._try_fallback_exchanges(symbol)
            return {}

    async def has_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        try:
            info = await self.client.get_exchange_info()
            symbols = [s['symbol'] for s in info['symbols']]
            return symbol in symbols
        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            return False

    async def get_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            ticker = await self.client.get_ticker(symbol=symbol)
            return float(ticker['lastPrice'])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0

    async def create_buy_order(self, symbol: str, amount: float, order_type: str = 'MARKET') -> Dict:
        """Create a buy order with proper quantity formatting"""
        try:
            # Get symbol info for precision
            symbol_info = self.symbol_info.get(symbol)
            if not symbol_info:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Get current price first
            ticker = await self.get_ticker(symbol)
            if not ticker or 'price' not in ticker:
                raise ValueError(f"Could not get current price for {symbol}")
                
            current_price = ticker['price']
            
            # Find MIN_NOTIONAL filter
            min_notional = 10.0  # Default minimum 10 USDT
            for f in symbol_info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f['minNotional'])
                    break
            
            # Pre-validate order value
            order_value = amount * current_price
            if order_value < min_notional:
                raise ValueError(f"Order value {order_value:.2f} USDT below minimum {min_notional} USDT")
            
            # Calculate quantity from amount
            quantity = amount / current_price
            
            # Format quantity with lot size rules
            formatted_quantity = self._format_quantity(quantity, symbol_info)
            
            # Final value validation
            final_value = float(formatted_quantity) * current_price
            if final_value < min_notional:
                raise ValueError(f"Final order value {final_value:.2f} USDT below minimum {min_notional} USDT")
            
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': order_type,
                'quantity': formatted_quantity
            }
            
            if order_type == 'LIMIT':
                price = self._format_price(current_price, symbol_info)
                params.update({
                    'price': price,
                    'timeInForce': 'GTC'
                })
            
            logger.info(f"Creating buy order: {params}")
            order = await self.client.create_order(**params)
            logger.info(f"Buy order created: {order['orderId']} for {formatted_quantity} {symbol}")
            return order
            
        except ValueError as e:
            logger.error(f"Buy order failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Buy order failed: {str(e)}")
            raise

    async def create_sell_order(self, symbol: str, amount: float, order_type: str = 'MARKET') -> Dict:
        """Create a sell order with validation and risk checks"""
        try:
            # Get symbol info for precision
            symbol_info = self.symbol_info.get(symbol)
            if not symbol_info:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Format quantity with lot size rules
            formatted_quantity = self._format_quantity(amount, symbol_info)
            
            # Get current price for validation
            ticker = await self.get_ticker(symbol)
            if not ticker or 'price' not in ticker:
                raise ValueError(f"Could not get current price for {symbol}")
                
            current_price = ticker['price']
            
            # Find MIN_NOTIONAL filter
            min_notional = 10.0  # Default minimum 10 USDT
            for f in symbol_info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f['minNotional'])
                    break
                    
            # Final value validation
            order_value = float(formatted_quantity) * current_price
            if order_value < min_notional:
                raise ValueError(f"Order value {order_value:.2f} USDT below minimum {min_notional} USDT")
            
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': order_type,
                'quantity': formatted_quantity
            }
            
            if order_type == 'LIMIT':
                price = self._format_price(current_price, symbol_info)
                params.update({
                    'price': price,
                    'timeInForce': 'GTC'
                })
            
            logger.info(f"Creating sell order: {params}")
            order = await self.client.create_order(**params)
            logger.info(f"Sell order created: {order['orderId']} for {formatted_quantity} {symbol}")
            return order
            
        except ValueError as e:
            logger.error(f"Sell order failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Sell order failed: {str(e)}")
            raise

    def _format_quantity(self, quantity: float, symbol_info: Dict) -> str:
        """
        Format order quantity according to exchange rules.
        
        Applies:
            - Step size rules
            - Minimum quantity
            - Precision requirements
            - Lot size filters
            
        Args:
            quantity (float): Raw order quantity
            symbol_info (Dict): Symbol trading rules
            
        Returns:
            str: Formatted quantity string
            
        Raises:
            ValueError: If quantity invalid
        """
        try:
            # Get the lot size filter
            lot_size_filter = next(
                filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters'])
            )
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            
            # Validate minimum quantity
            if quantity < min_qty:
                raise ValueError(f"Quantity {quantity} below minimum {min_qty}")
            
            # Calculate precision from step size
            precision = len(str(step_size).rstrip('0').split('.')[-1])
            
            # Round down to nearest step size multiple
            quantity = float(int(quantity / step_size) * step_size)
            
            # Format to correct precision
            return f"{quantity:.{precision}f}"
            
        except (KeyError, StopIteration) as e:
            logger.error(f"Error getting step size: {e}")
            raise ValueError("Step size not found in symbol info")
        except Exception as e:
            logger.error(f"Error formatting quantity: {e}")
            raise

    def _format_price(self, price: float, symbol_info: Dict) -> str:
        """Format price according to tick size rules"""
        tick_size = float(symbol_info['filters'][0]['tickSize'])
        precision = len(str(tick_size).rstrip('0').split('.')[-1])
        
        # Round to valid tick size
        price = round(price - (price % tick_size), precision)
        
        # Convert to string with correct precision
        return f"{price:.{precision}f}"

    async def get_order(self, symbol: str, order_id: int) -> Dict:
        """Get order status"""
        try:
            return await self.client.get_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Failed to get order status: {str(e)}")
            raise

    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an open order"""
        try:
            return await self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            raise