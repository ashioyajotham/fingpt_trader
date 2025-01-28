import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

async def run_trading_system(config_path: str, verbose: bool = False):
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trading system
    from main import TradingSystem
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    asyncio.run(run_trading_system(args.config, args.verbose))

if __name__ == "__main__":
    main()
