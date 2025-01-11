import os
import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import asyncio

from services.data_feeds.market_data_service import MarketDataService


async def monitor_prices():
    service = MarketDataService({"symbols": ["BTC/USDT", "ETH/USDT"], "interval": "1m"})

    await service.start()

    try:
        while True:
            quotes = await service.get_realtime_quote(["BTC/USDT", "ETH/USDT"])
            print(f"Current prices: {quotes}")
            await asyncio.sleep(60)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(monitor_prices())
