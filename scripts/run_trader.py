import asyncio
import argparse
from pathlib import Path
import sys

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from main import TradingSystem

async def run_trading_system(config_path: str):
    """Run the trading system with the specified configuration."""
    system = TradingSystem(config_path=config_path)
    
    try:
        await system.initialize()
        while True:
            market_data = await system.get_market_data()
            signals = await system.detect_inefficiencies(market_data)
            trades = system.generate_trades(signals)
            await system.execute_trades(trades)
            await asyncio.sleep(1)  # Adjust timing as needed
    except KeyboardInterrupt:
        print("\nShutting down trading system...")
    finally:
        await system.shutdown()

def main():
    parser = argparse.ArgumentParser(description='Run the FinGPT Trading System')
    parser.add_argument('--config', type=str, default='config/trading.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    asyncio.run(run_trading_system(args.config))

if __name__ == "__main__":
    main()
