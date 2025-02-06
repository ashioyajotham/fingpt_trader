import argparse
import asyncio
import logging
import signal
from pathlib import Path
import sys
from typing import Dict
import yaml
import platform

import os

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.trading.robo_service import RoboService
from services.exchanges.binance import BinanceClient

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.exchange = None
        self.robo_service = None

    async def startup(self):
        try:
            logger.debug("Starting trading system initialization...")
            logger.debug(f"Python version: {sys.version}")
            logger.debug(f"Platform: {platform.platform()}")
            
            # Initialize exchange
            logger.debug("Initializing exchange client...")
            self.exchange = await BinanceClient.create(self.config['exchange'])
            
            # Get trading pairs
            pairs = await self.exchange.get_trading_pairs()
            logger.debug(f"Available trading pairs: {len(pairs)}")
            
            # Initialize robo service
            logger.debug("Initializing RoboService...")
            self.robo_service = RoboService(self.config)
            await self.robo_service._setup()
            
            self.running = True
            logger.info("Trading system started successfully")
            
            # Log additional system info in verbose mode
            logger.debug(f"Trading mode: {'Test' if self.config['exchange'].get('test_mode') else 'Live'}")
            logger.debug(f"Active strategies: {list(self.config.get('strategies', {}).keys())}")
            logger.debug(f"Memory usage: {self._get_memory_usage():.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            await self.shutdown()
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    async def shutdown(self):
        self.running = False
        try:
            tasks = []
            
            if self.robo_service:
                tasks.append(self.robo_service.cleanup())
            if self.exchange:
                tasks.append(self.exchange.cleanup())
                
            # Wait for all cleanup tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Final wait to ensure all connections close
            await asyncio.sleep(0.5)
            
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise  # Re-raise to ensure proper process termination

    async def run(self):
        try:
            await self.startup()
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()

def load_config() -> Dict:
    """Load configuration from yaml file with proper path resolution"""
    try:
        args = parse_args()
        
        # If config path is provided via command line
        if args.config:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                # Make path absolute relative to project root
                config_path = Path(__file__).parent.parent / config_path
        else:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'trading.yaml'
        
        # Check if config exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        logger.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)
                
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='FinGPT Trading System')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def configure_windows_event_loop():
    """Configure event loop policy for Windows"""
    if platform.system() == 'Windows':
        try:
            # Import SelectorEventLoop explicitly
            from asyncio import SelectorEventLoop, WindowsSelectorEventLoopPolicy
            # Set policy to use SelectorEventLoop
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
            # Create and set the event loop
            loop = SelectorEventLoop()
            asyncio.set_event_loop(loop)
        except ImportError:
            logger.warning("Could not import SelectorEventLoop, falling back to ProactorEventLoop")
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Disable asyncio debug to prevent event loop warnings
        logging.getLogger('asyncio').setLevel(logging.INFO)

async def main():
    """Main entry point"""
    # Configure logging first
    logging.basicConfig(
        level=logging.DEBUG if '--verbose' in sys.argv else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        config = load_config()
        if not config:
            raise ValueError("Empty configuration loaded")
            
        trading_system = TradingSystem(config)
        await trading_system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    configure_windows_event_loop()
    asyncio.run(main())
