import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.data_feeds.market_data_service import MarketDataService
from services.robo.advisor_service import RoboAdvisorService


async def run_portfolio_optimization():
    # Initialize services
    advisor = RoboAdvisorService(
        {
            "risk_profile": "moderate",
            "rebalance_threshold": 0.05,
            "max_position_size": 0.2,
        }
    )

    market_data = MarketDataService(
        {"symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"], "interval": "1d"}
    )

    await advisor._setup()
    await market_data.start()

    try:
        # Get historical returns
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # Sample returns data
        returns = np.array(
            [
                [0.01, 0.02, 0.015],  # BTC returns
                [0.02, 0.01, 0.01],  # ETH returns
                [0.015, 0.025, 0.02],  # BNB returns
            ]
        )

        # Get optimal portfolio
        result = await advisor.optimize_portfolio(returns)

        print("\n=== Portfolio Optimization Results ===")
        print(f"Optimal Weights: {result['weights']}")
        print(f"Expected Return: {result['expected_return']:.2%}")
        print(f"Portfolio Volatility: {result['volatility']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe']:.2f}")

        # Example portfolio rebalancing
        current_portfolio = {"BTC/USDT": 0.4, "ETH/USDT": 0.35, "BNB/USDT": 0.25}

        print("\n=== Current vs Target Allocation ===")
        for asset, target in zip(
            ["BTC/USDT", "ETH/USDT", "BNB/USDT"], result["weights"]
        ):
            current = current_portfolio[asset]
            diff = target - current
            print(f"{asset}:")
            print(f"  Current: {current:.1%}")
            print(f"  Target:  {target:.1%}")
            print(f"  Action:  {'BUY' if diff > 0 else 'SELL'} {abs(diff):.1%}")

    finally:
        await market_data.stop()


if __name__ == "__main__":
    asyncio.run(run_portfolio_optimization())
