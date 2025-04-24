"""
FinGPT Trading System - Production Entry Point

A comprehensive algorithmic trading system leveraging FinGPT for sentiment analysis
and market inefficiency detection.

Architecture:
    1. Core Components:
        - Market Analysis Module:
            * Real-time market data processing
            * Technical analysis indicators
            * Market inefficiency detection
            * Sentiment analysis using FinGPT
            * Price prediction models
            
        - Portfolio Management:
            * Dynamic position sizing
            * Portfolio optimization
            * Risk-adjusted allocation
            * Multi-asset rebalancing
            
        - Risk Management:
            * Real-time risk monitoring
            * Position exposure limits
            * Value at Risk (VaR) calculations
            * Dynamic stop-loss management
            * Market regime detection
            
        - Execution Engine:
            * Smart order routing
            * Multiple exchange support
            * Order book analysis
            * Transaction cost analysis
            * Execution algorithm selection
            
        - Data Pipeline:
            * Real-time market data feeds
            * News and social media integration
            * Data normalization
            * Feature engineering
            * Historical data management

Configuration:
    The system requires proper setup of configuration files and environment:
    
    1. Environment Variables:
        - BINANCE_API_KEY: Exchange API credentials
        - BINANCE_API_SECRET: Exchange secret key
        - HUGGINGFACE_TOKEN: For FinGPT model access
        - CRYPTOPANIC_API_KEY: For crypto news feed
        - NEWS_API_KEY: Backup news source
        
    2. Configuration Files:
        - config/
            ├── trading.yaml: Main trading parameters
            ├── model.yaml: ML model configurations
            └── services.yaml: Service configurations
            
    3. Model Files:
        - models/
            ├── sentiment/: Sentiment analysis models
            └── market/: Market analysis models

      
    External:
        - Binance API
        - CryptoPanic API
        - HuggingFace API
        - NewsAPI
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import logging


import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import signal

from utils.logging import LogManager
from utils.config import ConfigManager
from services.data_feeds.news_service import NewsService
from services.data_feeds.market_data_service import MarketDataService
from models.llm.fingpt import FinGPT

# Replace existing logging config
LogManager({
    "log_dir": "logs",
    "level": "INFO",
}).setup_basic_logging()
logger = logging.getLogger(__name__)

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.market.inefficiency import MarketInefficencyDetector
from strategies.sentiment.analyzer import SentimentAnalyzer
from models.portfolio.optimization import PortfolioOptimizer
from models.portfolio.risk import RiskManager
from services.trading.robo_service import RoboService

class TradingSystem:
    """
    Production-grade algorithmic trading system with ML-powered analysis.
    This class orchestrates all components of the trading system including:
    market analysis, portfolio management, risk management, and trade execution.
    Key Features:
        - Real-time market data processing
        - ML-based sentiment analysis
        - Dynamic portfolio optimization
        - Risk-aware trade execution
        - Multi-exchange support
    Components:
        - MarketInefficiencyDetector: Identifies trading opportunities
        - SentimentAnalyzer: NLP-based market sentiment analysis
        - PortfolioOptimizer: Portfolio allocation and rebalancing
        - RiskManager: Risk monitoring and exposure management
        - RoboService: Automated portfolio management
    Attributes:
        config (Dict): System configuration parameters
        market_detector (MarketInefficiencyDetector): Market analysis
        sentiment_analyzer (SentimentAnalyzer): Sentiment analysis
        portfolio_optimizer (PortfolioOptimizer): Portfolio management
        risk_manager (RiskManager): Risk monitoring
        robo_service (RoboService): Automated trading
        exchange_clients (Dict[str, ExchangeClient]): Exchange connections
        client_profiles (Dict[str, Dict]): Client configurations
    Configuration Parameters:
        trading:
            pairs (List[str]): Trading pairs to monitor
            initial_balance (float): Starting portfolio balance
            confidence_threshold (float): Min confidence for trades
            min_trade_amount (float): Minimum trade size
            max_position_size (float): Maximum position size
        risk:
            max_drawdown (float): Maximum allowed drawdown
            position_limit (float): Maximum position size
            var_limit (float): Value at Risk limit
            leverage_limit (float): Maximum leverage
        model:
            fingpt_config (Dict): FinGPT model parameters
            market_config (Dict): Market analysis parameters
    Methods:
        initialize(): Setup system components
        run(): Main trading loop
        shutdown(): Cleanup resources
        execute_trades(): Execute trading decisions
        update_portfolio(): Update portfolio states
        check_risk_metrics(): Monitor risk limits
    Example:
        >>> system = TradingSystem("config/trading.yaml")
        >>> asyncio.run(system.run())
    """
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config('trading')
        if not self.config:
            raise ValueError("Trading configuration not found")
        # Load and validate environment variables
        load_dotenv(override=True, verbose=True)
        # Enhanced environment variable logging
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        logger.info(f"API Key present: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
        logger.info(f"API Secret present: {bool(api_secret)} (length: {len(api_secret) if api_secret else 0})")
        # Verify required environment variables
        required_env = {
            'BINANCE_API_KEY': 'Binance API key',
            'BINANCE_API_SECRET': 'Binance API secret'
        }
        missing = []
        for var, name in required_env.items():
            value = os.getenv(var)
            if not value or len(value.strip()) < 10:  # Basic validation
                missing.append(name)
        if missing:
            raise ValueError(f"Missing or invalid {', '.join(missing)}")
        logger.info("Environment variables validated successfully")
        # Load and process config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Load model and service configs first
        with open(self.config['configs']['model']) as f:
            self.model_config = yaml.safe_load(f)
        with open(self.config['configs']['services']) as f:
            self.service_config = yaml.safe_load(f)
        # Initialize FinGPT model first
        self.fingpt_model = FinGPT(self.model_config['fingpt'])
        # Initialize data services
        self.market_data_service = MarketDataService(self.service_config['data_feeds']['market'])
        self.news_service = NewsService(self.service_config['data_feeds']['news'])
        # Initialize sentiment analyzer with model instance
        sentiment_config = {
            **self.config.get('sentiment', {}),
            'model': self.fingpt_model,
            'data_feeds': self.service_config['data_feeds']
        }
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
        self.market_detector = MarketInefficencyDetector(self.config.get('market', {}))
        self.portfolio_optimizer = PortfolioOptimizer(self.config.get('portfolio', {}))
        self.risk_manager = RiskManager(
            max_drawdown=self.config.get('risk', {}).get('max_drawdown', 0.1),
            var_limit=self.config.get('risk', {}).get('var_limit', 0.02)
        )
        self.portfolio = None
        self.market_state = {}
        self.exchange_clients = {}
        self.is_running = False
        # Add robo advisor components
        self.robo_service = RoboService(self.config.get('robo', {}))
        self.client_profiles = {}

    async def initialize(self):
        """Initialize the trading system components"""
        logger.info("Starting trading system initialization...")
        
        try:
            # Set up exchange connections
            from services.exchanges.binance import BinanceClient
            
            # Initialize exchange clients
            for exchange_config in self.config.get('exchanges', []):
                exchange_name = exchange_config.get('name', '').lower()
                if exchange_name == 'binance':
                    # Use the create factory method instead of direct constructor
                    self.exchange_clients[exchange_name] = await BinanceClient.create({
                        'api_key': os.getenv('BINANCE_API_KEY'),
                        'api_secret': os.getenv('BINANCE_API_SECRET'),
                        'test_mode': exchange_config.get('test_mode', True),
                        'options': exchange_config.get('options', {})
                    })
                    
            # Initialize robo service
            await self.robo_service.setup(
                exchange_clients=self.exchange_clients,
                trading_pairs=self.config.get('trading', {}).get('pairs', ['BTCUSDT']),
                initial_balance=self.config.get('trading', {}).get('initial_balance', 10000.0)
            )
            
            logger.info("Trading system initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    # Add this method to convert order size properly
    def _format_order_size(self, trade: Dict) -> float:
        """Format order size correctly for the exchange"""
        try:
            exchange = trade['exchange']
            symbol = trade['symbol']
            client = self.exchange_clients[exchange]
            
            # Get symbol info/limits from exchange
            symbol_info = client.get_symbol_info(symbol)
            
            # Extract minimum quantity and precision
            min_qty = float(symbol_info.get('min_qty', 0.00001))
            qty_precision = int(symbol_info.get('qty_precision', 5))
            
            # Round to appropriate precision 
            formatted_qty = round(max(min_qty, float(trade['size'])), qty_precision)
            
            logger.info(f"Formatted {trade['size']} to {formatted_qty} for {symbol} " 
                       f"(min: {min_qty}, precision: {qty_precision})")
            
            return formatted_qty
            
        except Exception as e:
            logger.error(f"Error formatting order size: {str(e)}")
            # Return original size as fallback
            return float(trade['size'])

    def _sanitize_text(self, text: str) -> str:
        """Remove model output patterns from input text"""
        import re
        
        if not text:
            return ""
        
        # First remove entire lines with model outputs
        clean_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if any(pattern in line.lower() for pattern in [
                'sentiment score:', 'sentiment:', 'score:', 
                'you:', 'assistant:', 'user:',
                'analysis:', 'raw response', 'thank you for your time',
                'i don\'t understand', 'prediction:', 'confidence:',
                'rating:', 'bullish:', 'bearish:'  # Additional financial markers
            ]):
                continue
            clean_lines.append(line)
        
        # Join cleaned lines
        cleaned_text = " ".join(clean_lines)
        
        # Remove specific patterns that could appear inline
        patterns_to_remove = [
            r'sentiment score:[\s\-\.x0-9]+',
            r'sentiment:[\s\-\.x0-9]+', 
            r'you:.*?assistant:',
            r'analysis:.*?raw response',
            r'thank you for your time',
            r'i don\'t understand',
            r'prediction:[\s\-\.x0-9]+',
            r'confidence:[\s\-\.x0-9]+',
            r'[<\[].*?[>\]]'  # Remove angle/square bracket tags
        ]
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    async def shutdown(self):
        """Clean up all system resources"""
        # Prevent multiple shutdown calls
        if hasattr(self, '_shutting_down') and self._shutting_down:
            logger.info("Shutdown already in progress, skipping")
            return
            
        self._shutting_down = True
        logger.info("\nShutting down trading system...")
        
        try:
            # Close exchange connections
            if hasattr(self, 'exchange_clients'):
                for client in self.exchange_clients.values():
                    await client.close_connections()
                
            # Stop all services in sequence
            if hasattr(self, 'market_data_service'):
                await self.market_data_service.stop()
                
            if hasattr(self, 'news_service'):
                await self.news_service.stop()
                
            if hasattr(self, 'sentiment_analyzer'):
                await self.sentiment_analyzer.cleanup()
                
            if hasattr(self, 'market_detector'):
                await self.market_detector.cleanup()
                
            if hasattr(self, 'portfolio_optimizer'):
                if hasattr(self.portfolio_optimizer, 'cleanup'):
                    await self.portfolio_optimizer.cleanup()
                else:
                    logger.debug("Portfolio optimizer has no cleanup method, skipping")
                
            if hasattr(self, 'robo_service'):
                await self.robo_service.cleanup()
                
            logger.info("Trading system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Helper function to wait for shutdown signal
async def wait_for_shutdown(shutdown_event):
    await shutdown_event.wait()
    logger.info("Shutdown event triggered")

async def main():
    """Main entry point for the trading system."""
    system = None
    try:
        # Create and initialize the trading system
        system = TradingSystem("config/trading.yaml")
        await system.initialize()
        
        # Run the main system loop
        await system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        # Ensure proper cleanup even if there's an error
        if system:
            await system.shutdown()

if __name__ == "__main__":
    # Configure Windows-specific event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Set up signal handling before starting the loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create a Future to track shutdown requests
    shutdown_event = asyncio.Event()
    
    # Define signal handlers outside the loop
    def handle_shutdown(sig, frame):
        logger.info("\nShutdown signal received...")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        # Run main async function and monitor for shutdown signal
        main_task = loop.create_task(main())
        
        # Create a task from the wait_for_shutdown coroutine
        shutdown_task = loop.create_task(wait_for_shutdown(shutdown_event))
        
        # Run until either main completes or shutdown is requested
        try:
            loop.run_until_complete(
                asyncio.wait([
                    main_task,
                    shutdown_task  # Now using a task instead of raw coroutine
                ], return_when=asyncio.FIRST_COMPLETED)
            )
        finally:
            # Set shutdown event when main task completes
            shutdown_event.set()
            # Wait briefly for other task to respond to shutdown signal
            loop.run_until_complete(asyncio.sleep(0.5))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Top-level exception: {e}")
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
        
        # Wait briefly for cancellations to process
        if pending:
            loop.run_until_complete(asyncio.wait(pending, timeout=1.0))
        
        # Close the loop
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("AsyncIO event loop closed")
        
        # Force exit if still hanging
        import os
        os._exit(0)  # Force exit the process

