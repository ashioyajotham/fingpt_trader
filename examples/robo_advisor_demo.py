"""
Robo Advisor Portfolio Optimization Demo

This example demonstrates how to use the RoboService component to:
1. Generate portfolio optimization recommendations
2. Analyze individual assets for trading signals
3. Calculate key portfolio metrics (return, volatility, Sharpe ratio)
4. Determine portfolio rebalancing actions

The script simulates a year of historical returns and generates
a recommended portfolio allocation with actionable trade suggestions.

Usage:
    python examples/robo_advisor_demo.py
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")  # Explicitly provide path to root .env file

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

# Windows-specific event loop policy to address known issues with asyncio on Windows
# This is required for aiodns and other libraries that need SelectorEventLoop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from services.data_feeds.market_data_service import MarketDataService
from services.trading.robo_service import RoboService


async def run_portfolio_optimization():
    # Initialize services
    advisor = RoboService(
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
        assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Note: symbols format matching your service
        result = {"weights": [], "signals": {}}

        for i, asset in enumerate(assets):
            # Get current market price (you might need to get this from market_data)
            current_price = 50000 if "BTC" in asset else 3000 if "ETH" in asset else 400  # Example values
            
            # Analyze each position
            signal = await advisor.analyze_position(asset, current_price)
            result["signals"][asset] = signal
            
            # Use the returns matrix to estimate target weights
            # This is a simplified approach - not real portfolio optimization
            weight = np.mean(returns[i]) / np.sum([np.mean(r) for r in returns])
            result["weights"].append(weight)

        # Convert weights list to numpy array for calculations
        weights_array = np.array(result["weights"])

        # Calculate metrics using the numpy array
        result["expected_return"] = np.sum(weights_array * np.mean(returns, axis=1))
        result["volatility"] = np.sqrt(np.sum(weights_array**2 * np.var(returns, axis=1)))
        result["sharpe"] = result["expected_return"] / result["volatility"] if result["volatility"] > 0 else 0

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
