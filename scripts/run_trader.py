"""
FinGPT Trading System - Development Runner

A simplified runner script for development and testing purposes.
Provides basic trading system functionality with focus on:
- Single exchange testing (Binance)
- Basic robo-advisory features
- Strategy validation
- System component testing

This script is NOT intended for production use.
For production deployment, use main.py in the root directory.

Key Components:
    - TradingSystem: Simplified system orchestrator
    - BinanceClient: Single exchange support
    - RoboService: Basic robo-advisory features

Usage:
    python scripts/run_trader.py [--config CONFIG_PATH] [--verbose]
    
Development Tools:
    - Verbose logging for debugging
    - Test mode by default
    - Simplified component initialization
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
from dotenv import load_dotenv  # Add this import
import time
import datetime

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.trading.robo_service import RoboService
from services.exchanges.binance import BinanceClient
from utils.logging import LogManager

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
        self.last_status_update = 0
        self.status_interval = 60  # Status update every 60 seconds
        self.monitored_pairs = ['BTCUSDT', 'ETHUSDT']  # Default pairs to monitor
        self.price_data = {}

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

    async def update_market_data(self):
        """Update price data for monitored pairs"""
        try:
            for pair in self.monitored_pairs:
                ticker = await self.exchange.get_ticker(pair)
                self.price_data[pair] = {
                    'price': float(ticker['lastPrice']),
                    'change_24h': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume'])
                }
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")

    async def print_status_update(self):
        """Print periodic status update"""
        now = time.time()
        if now - self.last_status_update >= self.status_interval:
            self.last_status_update = now
            logger.info("\n=== Status Update ===")
            logger.info(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Market Data
            for pair, data in self.price_data.items():
                logger.info(f"\n{pair}:")
                logger.info(f"  Price: ${data['price']:.2f}")
                logger.info(f"  24h Change: {data['change_24h']:.2f}%")
                logger.info(f"  Volume: {data['volume']:.2f}")
            
            # System Status
            logger.info(f"\nSystem Status:")
            logger.info(f"  Running Time: {time.time() - self.start_time:.1f}s")
            logger.info(f"  Memory Usage: {self._get_memory_usage():.1f}MB")
            logger.info("==================\n")

    async def run(self):
        try:
            self.start_time = time.time()
            await self.startup()
            
            while self.running:
                # Update market data
                await self.update_market_data()
                
                # Print status update
                await self.print_status_update()
                
                # Main loop interval
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
        # Load environment variables first
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        print("\n=== FinGPT Trading System ===")
        print(f"✓ Environment loaded\n")
        
        # Security-conscious debug prints
        print("\nEnvironment Status:")
        print(f"BINANCE_API_KEY: {'✓ Present' if os.environ.get('BINANCE_API_KEY') else '✗ Missing'}")
        print(f"BINANCE_SECRET_KEY: {'✓ Present' if os.environ.get('BINANCE_SECRET_KEY') else '✗ Missing'}")
        print() # Empty line for readability
        
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
            # Load YAML with environment variable substitution
            config_text = f.read()
            # Replace SECRET_KEY with API_SECRET to match yaml expectations
            config_text = config_text.replace('${BINANCE_API_SECRET}', os.environ.get('BINANCE_SECRET_KEY', ''))
            config = yaml.safe_load(os.path.expandvars(config_text))
        
        # Debug prints
        print(f"Config loaded successfully")
        print(f"Exchange API Key configured: {bool(config['exchange']['api_key'])}")
        print(f"Exchange API Secret configured: {bool(config['exchange']['api_secret'])}")
        
        return config
                
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='FinGPT Trading System')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
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
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(console_handler)
    
    # Set log level based on verbosity
    log_level = "DEBUG" if '--verbose' in sys.argv else "INFO"
    logging.getLogger().setLevel(log_level)
    
    # Initialize file logging
    log_config = {
        "log_dir": "logs",
        "level": log_level
    }
    LogManager(log_config).setup_basic_logging()
    
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
