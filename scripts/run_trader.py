"""
FinGPT Trading System - Development Runner

A simplified runner script for development and testing purposes.
Provides basic trading system functionality with focus on:
- Single exchange testing (Binance)
- Basic robo-advisory features
- Strategy validation
- System component testing
- Trading pair limitations for faster testing

This script is NOT intended for production use.
For production deployment, use main.py in the root directory.

Key Components:
    - TradingSystem: Simplified system orchestrator
    - BinanceClient: Single exchange support
    - RoboService: Basic robo-advisory features

Usage:
    python scripts/run_trader.py [OPTIONS]

Options:
    --config CONFIG_PATH    Path to config file (default: config/trading.yaml)
    --verbose              Enable verbose logging
    --max-pairs N         Limit number of trading pairs for testing
    --pairs SYMBOLS       Specify trading pairs (comma-separated, e.g. BTCUSDT,ETHUSDT)
    
Development Features:
    - Verbose logging for debugging
    - Test mode by default
    - Simplified component initialization
    - Trading pair limitation options:
        1. Command line arguments (--max-pairs, --pairs)
        2. Development config in trading.yaml
        3. Full pair set if no limitations specified
"""

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
    """
    Main trading system orchestrator.
    
    Handles initialization, execution, and cleanup of trading components:
    - Exchange connections
    - Trading strategies
    - Portfolio management
    - Risk monitoring
    
    Attributes:
        config (Dict): System configuration
        running (bool): System running state
        exchange (BinanceClient): Exchange client instance
        robo_service (RoboService): Robo-advisory service instance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trading system with configuration.
        
        Args:
            config (Dict): Configuration dictionary from YAML
        """
        self.config = config
        self.running = False
        self.exchange = None
        self.robo_service = None

    async def startup(self):
        """
        Initialize system components in sequence.
        
        Startup sequence:
        1. Exchange client initialization
        2. Trading pairs discovery
        3. RoboService setup
        4. System state initialization
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            logger.debug("Starting trading system initialization...")
            logger.debug(f"Python version: {sys.version}")
            logger.debug(f"Platform: {platform.platform()}")
            
            # Initialize exchange
            logger.debug("Initializing exchange client...")
            self.exchange = await BinanceClient.create(self.config['exchange'])
            
            # Get trading pairs with development mode handling
            pairs = await self.exchange.get_trading_pairs()
            
            # Handle command line pair limitations first
            args = parse_args()
            if args.pairs:
                # Use specific pairs from command line
                requested_pairs = args.pairs.upper().split(',')
                pairs = [p for p in pairs if p in requested_pairs]
                logger.info(f"Using specified pairs: {pairs}")
            elif args.max_pairs:
                # Limit number of pairs from command line
                pairs = pairs[:args.max_pairs]
                logger.info(f"Limited to {len(pairs)} pairs")
            # Fall back to development config if no command line args
            elif self.config.get('development', {}).get('enabled'):
                dev_config = self.config['development']
                if 'test_pairs' in dev_config:
                    pairs = [p for p in pairs if p in dev_config['test_pairs']]
                elif 'max_pairs' in dev_config:
                    pairs = pairs[:dev_config['max_pairs']]
                logger.info(f"Development mode: Using {len(pairs)} pairs")
            
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
    """
    Load and validate system configuration.
    
    Searches for config file in the following order:
    1. Command line argument path
    2. Default path (config/trading.yaml)
    
    Returns:
        Dict: Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is empty or invalid
    """
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
    parser.add_argument('--max-pairs', type=int, help='Limit number of trading pairs')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of trading pairs (e.g. BTCUSDT,ETHUSDT)')
    return parser.parse_args()

def configure_windows_event_loop():
    """
    Configure event loop for Windows compatibility.
    
    Handles Windows-specific event loop requirements:
    - Uses SelectorEventLoop when possible
    - Falls back to ProactorEventLoop if necessary
    - Configures proper DNS resolution
    """
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
    """
    Main entry point for the trading system.
    
    Execution flow:
    1. Configure logging based on verbosity
    2. Load and validate configuration
    3. Initialize trading system
    4. Run main trading loop
    5. Handle graceful shutdown
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
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
