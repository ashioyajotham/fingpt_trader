"""
Market Data Service Example

This example demonstrates how to use the MarketDataService to:
1. Connect to cryptocurrency exchanges (Binance)
2. Retrieve real-time market data
3. Handle API differences through adaptive method calls
4. Implement proper async handling and cleanup

The script is designed to be resilient against API changes and 
provides diagnostic information when encountering issues.

Usage:
    python examples/market_data.py
"""
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')  # Explicitly provide path to root .env file

# Add project root to Python path to ensure imports work correctly
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import asyncio
import json

# Windows-specific event loop policy to address known issues with asyncio on Windows
# This is required for aiodns and other libraries that need SelectorEventLoop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import the MarketDataService which abstracts exchange API interactions
from services.data_feeds.market_data_service import MarketDataService

async def monitor_prices():
    """
    Main coroutine that initializes the MarketDataService and 
    continuously polls for current cryptocurrency prices.
    
    This function demonstrates:
    - Service initialization
    - Exchange connectivity
    - Multiple approaches to retrieve price data
    - Error handling and graceful shutdown
    """
    # Initialize MarketDataService with desired trading pairs and timeframe
    service = MarketDataService({"symbols": ["BTCUSDT", "ETHUSDT"], "interval": "1m"})

    print("Starting market data service...")
    await service.start()  # Connect to exchanges and initialize data feeds
    print("Service started successfully!")

    try:
        # Main monitoring loop
        while True:
            try:
                # Access the underlying exchange client to query prices
                # The client implementation may vary based on the exchange
                client = service.exchange
                
                # Approach 1: Try get_symbol_ticker method (Binance standard method)
                if hasattr(client, 'get_symbol_ticker'):
                    btc = await client.get_symbol_ticker(symbol="BTCUSDT")
                    eth = await client.get_symbol_ticker(symbol="ETHUSDT")
                    print(f"BTC: {btc}, ETH: {eth}")
                
                # Approach 2: Try get_ticker method (alternative API)
                elif hasattr(client, 'get_ticker'):
                    btc = await client.get_ticker(symbol="BTCUSDT")
                    eth = await client.get_ticker(symbol="ETHUSDT")
                    # Print abbreviated response to examine structure
                    print(f"BTC ticker: {json.dumps(btc, indent=2)[:100]}...")
                    print(f"ETH ticker: {json.dumps(eth, indent=2)[:100]}...")
                
                # Approach 3: Diagnostic - list available price-related methods
                else:
                    print("Available methods on client:")
                    methods = [m for m in dir(client) if not m.startswith('_') and 'price' in m.lower()]
                    print(methods)
            
            except Exception as e:
                # Catch and report any errors during price fetching
                print(f"Error fetching prices: {type(e).__name__}: {e}")
            
            # Wait before next update (5 seconds is appropriate for example purposes)
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        print("Monitoring stopped by user")
    finally:
        # Ensure proper cleanup regardless of exit reason
        if hasattr(service, 'stop'):
            await service.stop()


if __name__ == "__main__":
    try:
        # Start the async event loop with our main coroutine
        asyncio.run(monitor_prices())
    except Exception as e:
        # Catch any uncaught exceptions at the top level
        print(f"Error in market data example: {e}")
