"""
FinGPT Trading System - Main Entry Point

"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from datetime import datetime
import argparse

from utils.config import ConfigManager
from utils.logging import LogManager
from strategies.sentiment.analyzer import SentimentAnalyzer
from strategies.portfolio.manager import PortfolioManager
from services.trading.robo_service import RoboService

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FinGPT Trading System")
    
    # Verbosity options (mutually exclusive)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true",
                          help="Enable verbose logging (INFO level)")
    verbosity.add_argument("-d", "--debug", action="store_true",
                          help="Enable debug logging (DEBUG level)")
    verbosity.add_argument("-q", "--quiet", action="store_true",
                          help="Minimal logging (ERROR level)")
    
    return parser.parse_args()

# Initialize logging
LogManager({"log_dir": "logs", "level": "INFO"}).setup_basic_logging()
logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration path"""
        self.startup_time = datetime.now()
        self.is_running = False
        self.exchange_clients = {}
        
        # Load configuration
        self.config_manager = ConfigManager()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        if not self.config or 'exchanges' not in self.config:
            raise ValueError("Invalid trading configuration")
            
        logger.info("Trading system initialized with configuration")
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
        self.portfolio_manager = PortfolioManager(self.config)
        self.robo_service = RoboService(self.config)
        
        # Trading state
        self.market_data = {}
        self.active_streams = {}
        self.last_analysis = {}

    async def _setup_exchange(self, exchange_config: Dict) -> None:
        """Setup single exchange connection"""
        try:
            exchange_type = exchange_config.get('name', '').lower()
            
            if exchange_type == 'binance':
                from services.exchanges.binance import BinanceClient
                
                # Get validated credentials
                creds = self.config_manager.get_exchange_credentials('binance')
                
                # Create exchange client
                client_config = {
                    'api_key': creds['api_key'],
                    'api_secret': creds['api_secret'],
                    'test_mode': exchange_config.get('test_mode', True),
                    'options': exchange_config.get('options', {})
                }
                
                client = await BinanceClient.create(client_config)
                self.exchange_clients[exchange_type] = client
                logger.info(f"Successfully connected to {exchange_type}")
                
            else:
                logger.warning(f"Unsupported exchange type: {exchange_type}")
                
        except Exception as e:
            logger.error(f"Failed to setup {exchange_type}: {str(e)}")
            raise

    async def initialize(self) -> None:
        """Initialize all system components"""
        try:
            logger.info("Starting system initialization...")
            
            # Setup exchange connections
            for exchange_config in self.config['exchanges']:
                client = await self._setup_exchange(exchange_config)
                if client:
                    # Start market data streams for configured pairs
                    await client.start_market_streams(
                        self.config['trading']['pairs']
                    )
                    await client.start_data_processing()
            
            # Initialize services
            await self.robo_service._setup()
            await self.sentiment_analyzer.initialize()
            
            self.is_running = True
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.shutdown()
            raise

    async def run(self) -> None:
        """Main trading loop"""
        try:
            await self.initialize()
            
            while self.is_running:
                try:
                    # Process market data
                    for exchange, client in self.exchange_clients.items():
                        for pair in self.config['trading']['pairs']:
                            # Get market data
                            orderbook = client.get_orderbook_snapshot(pair)
                            
                            # Update sentiment - Remove await as method returns float
                            sentiment_impact = self.sentiment_analyzer.get_sentiment_impact(pair)
                            
                            # Update portfolio manager with correct structure
                            market_data = {
                                'orderbook': orderbook,
                                'sentiment': {pair: sentiment_impact},
                                'market': {
                                    'candles': self.market_data.get('candles', {}),
                                    'trades': self.market_data.get('trades', {})
                                }
                            }
                            
                            signals = await self.portfolio_manager.generate_signals(market_data)
                            
                            # Execute trades via robo service
                            if signals:
                                for signal in signals:
                                    await self.robo_service.analyze_position(
                                        pair,
                                        float(orderbook['asks'][0][0]),
                                        signal
                                    )
                    
                    # Wait for next iteration
                    await asyncio.sleep(
                        self.config.get('trading', {}).get('loop_interval', 60)
                    )
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Clean shutdown of all components"""
        self.is_running = False
        
        try:
            # Cleanup services
            await self.robo_service.cleanup()
            await self.sentiment_analyzer.cleanup()
            
            # Cleanup exchange connections
            for name, client in self.exchange_clients.items():
                try:
                    await client.cleanup()
                    logger.info(f"Cleaned up {name} connection")
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            logger.info("System shutdown complete")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Determine logging level
    if args.debug:
        log_level = "DEBUG"
    elif args.verbose:
        log_level = "INFO"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = "WARNING"  # Default level
    
    # Initialize logging with verbosity
    LogManager({"log_dir": "logs"}).setup_basic_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Windows-specific settings
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create and run system
    system = TradingSystem("config/trading.yaml")
    
    try:
        asyncio.run(system.run())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)