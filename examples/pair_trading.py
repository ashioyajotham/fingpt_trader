import os
import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

import asyncio

from strategies.inefficiency.inefficiency_strategy import InefficiencyStrategy


async def find_pairs():
    strategy = InefficiencyStrategy(
        {"window": 20, "z_threshold": 2.0, "pairs": [["BTC/USDT", "ETH/USDT"]]}
    )

    signals = await strategy.generate_signals()
    print(f"Trading signals: {signals}")


if __name__ == "__main__":
    asyncio.run(find_pairs())
