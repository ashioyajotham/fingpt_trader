"""
FinGPT Trading System - Main Entry Point

"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import os
from datetime import datetime
import argparse
from dotenv import load_dotenv  # Add this import

# Load environment variables first
load_dotenv()

from utils.config import ConfigManager
from utils.logging import LogManager
from strategies.sentiment.analyzer import SentimentAnalyzer
from strategies.portfolio.manager import PortfolioManager
from services.trading.robo_service import RoboService
from services.data_feeds.news_service import NewsService, NewsDataFeed
from services.data_feeds.market_data_service import MarketDataService, MarketDataFeed
from models.llm.fingpt import FinGPT  # Update import to use existing model

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
        # Ensure environment variables are loaded
        if not os.getenv('HUGGINGFACE_TOKEN'):
            # Try loading from specific .env file if exists
            env_path = Path(__file__).parent / '.env'
            if (env_path).exists():
                load_dotenv(env_path)
            else:
                raise ValueError("HUGGINGFACE_TOKEN not found in environment or .env file")

        self.startup_time = datetime.now()
        self.is_running = False
        self.exchange_clients = {}
        
        # Load trading config first
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Load model and service configs
        with open(self.config['configs']['model']) as f:
            self.model_config = yaml.safe_load(f)
            
        with open(self.config['configs']['services']) as f:
            self.service_config = yaml.safe_load(f)
        
        if not self.config or 'exchanges' not in self.config:
            raise ValueError("Invalid trading configuration")
            
        logger.info("Trading system initialized with configuration")
        
        # Initialize FinGPT model first
        self.fingpt_model = FinGPT(self.model_config['fingpt'])
        
        # Initialize data services
        self.market_data_service = MarketDataService(self.service_config['data_feeds']['market'])
        self.news_service = NewsService(self.service_config['data_feeds']['news'])
        
        # Initialize sentiment analyzer with model instance
        sentiment_config = {
            **self.config.get('sentiment', {}),
            'model': self.fingpt_model,  # Pass model instance
            'data_feeds': self.service_config['data_feeds']
        }
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
        
        # Initialize other components
        self.portfolio_manager = PortfolioManager(self.config)
        self.robo_service = RoboService(self.config)
        
        # Trading state
        self.market_data = {}
        self.active_streams = {}
        self.last_analysis = {}

        # Data feed configurations
        market_config = {
            'pairs': self.config['trading']['pairs'],
            'update_interval': self.config.get('market_data', {}).get('update_interval', 60),
            'cache_size': self.config.get('market_data', {}).get('cache_size', 1000)
        }
        
        news_config = {
            'update_interval': self.config.get('news', {}).get('update_interval', 300),
            'sources': self.config.get('news', {}).get('sources', []),
            'keywords': self.config.get('news', {}).get('keywords', {})
        }
        
        # Initialize services with proper configs
        self.market_feed = MarketDataFeed(market_config)
        self.news_feed = NewsDataFeed(news_config)

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
            
            # Initialize in correct order
            await self.fingpt_model.initialize()
            await self.market_data_service.start()
            await self.news_service.start()
            await self.sentiment_analyzer.initialize()
            
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
            await self.portfolio_manager.initialize()
            
            self.is_running = True
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.shutdown()
            raise

    async def run(self) -> None:
        """Enhanced main trading loop with data feeds"""
        try:
            await self.initialize()
            
            while self.is_running:
                try:
                    # Update market data
                    for pair in self.config['trading']['pairs']:
                        market_data = await self.market_feed.get_latest(pair)
                        news_data = await self.news_feed.get_latest()
                        
                        # Process market data
                        if market_data.get('orderbook'):
                            # Update sentiment with market context
                            sentiment_impact = self.sentiment_analyzer.get_sentiment_impact(pair)
                            
                            # Process relevant news
                            for news_item in news_data:
                                if pair in news_item.get('symbols', []):
                                    await self.sentiment_analyzer.add_sentiment_data(
                                        pair,
                                        news_item['content'],
                                        news_item['timestamp']
                                    )
                            
                            # Prepare market context
                            market_context = {
                                'orderbook': market_data['orderbook'],
                                'trades': market_data['trades'],
                                'sentiment': {pair: sentiment_impact},
                                'news': [n for n in news_data if pair in n.get('symbols', [])]
                            }
                            
                            # Generate trading signals
                            signals = await self.portfolio_manager.generate_signals(market_context)
                            
                            # Execute trades
                            if signals:
                                for signal in signals:
                                    best_ask = float(market_data['orderbook']['asks'][0][0])
                                    await self.robo_service.analyze_position(pair, best_ask, signal)
                    
                    # Controlled delay between iterations
                    await asyncio.sleep(self.config.get('trading', {}).get('loop_interval', 60))
                    
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
            # Cleanup in reverse order
            await self.fingpt_model.cleanup()  # Clean up FinGPT first
            # Stop data feeds first
            await self.market_data_service.stop()
            await self.news_service.stop()
            
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