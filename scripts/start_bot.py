import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.data_feeds.market_data_service import MarketDataService
from services.data_feeds.news_service import NewsService
from services.trading.broker_service import BrokerService
from strategies.hybrid.hybrid_strategy import HybridStrategy
from utils.config import ConfigManager
from utils.logging import LogManager

async def main():
    # Load configs
    load_dotenv()
    config = ConfigManager().get_config('trading')
    LogManager(config.get('logging', {}))
    
    # Initialize services
    market_data = MarketDataService(config.get('market_data'))
    news_service = NewsService(config.get('news'))
    broker = BrokerService(config.get('broker'))
    
    # Initialize strategy
    strategy = HybridStrategy(config.get('strategy'))
    
    try:
        # Start services
        await market_data._setup()
        await news_service._setup()
        await broker._setup()
        
        # Start trading loop
        while True:
            # Get market data
            quotes = await market_data.get_realtime_quote(config['symbols'])
            
            # Get news
            news = await news_service.get_news(" OR ".join(config['symbols']))
            
            # Generate signals
            signals = await strategy.generate_signals()
            
            # Execute trades
            if signals:
                for signal in signals:
                    await broker.submit_order(signal)
                    
            await asyncio.sleep(config.get('interval', 60))
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await market_data._cleanup()
        await news_service._cleanup()
        await broker._cleanup()

if __name__ == "__main__":
    print("Starting FinGPT Trader...")
    asyncio.run(main())