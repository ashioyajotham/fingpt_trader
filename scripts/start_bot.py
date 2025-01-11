import asyncio
import platform
import sys
from pathlib import Path
from services.data_feeds.market_data_service import MarketDataService
from services.data_feeds.news_service import NewsService
from services.trading.broker_service import BrokerService
from strategies.hybrid.hybrid_strategy import HybridStrategy
from utils.config import ConfigManager
from utils.logging import LogManager


from dotenv import load_dotenv

# Configure event loop policy for Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)


def parse_interval(interval_str: str) -> int:
    """Convert time interval string to seconds"""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}

    if isinstance(interval_str, (int, float)):
        return int(interval_str)

    unit = interval_str[-1].lower()
    value = int(interval_str[:-1])

    if unit not in units:
        raise ValueError(f"Invalid interval unit: {unit}")

    return value * units[unit]


async def main():
    # Load configs
    load_dotenv()
    config = ConfigManager().get_config("trading") or {}
    logger = LogManager(config.get("logging", {}))

    # Validate required config
    market_config = config.get("market_data", {})
    symbols = market_config.get("symbols", ["BTC/USDT", "ETH/USDT"])

    # Convert interval to seconds
    interval = parse_interval(market_config.get("interval", "60s"))

    if not symbols:
        raise ValueError("No trading symbols configured")

    # Initialize services with validated config
    market_data = MarketDataService(market_config)
    news_service = NewsService(config.get("news", {}))
    broker = BrokerService(config.get("broker", {}))
    strategy = HybridStrategy(config.get("strategy"))

    # Start services
    await market_data.start()
    await news_service.start()
    await broker.start()

    try:
        while True:
            quotes = await market_data.get_realtime_quote(symbols)
            news = await news_service.get_news(" OR ".join(symbols))
            signals = await strategy.generate_signals()

            if signals:
                for signal in signals:
                    await broker.submit_order(signal)

            await asyncio.sleep(interval)  # Use converted integer interval

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await market_data.stop()
        await news_service.stop()
        await broker.stop()


if __name__ == "__main__":
    print("Starting FinGPT Trader...")
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
