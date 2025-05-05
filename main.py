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

# Import section - add these imports at the top
import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path

# Add Windows-specific event loop policy fix before using asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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
    """
    def __init__(self):
        """Initialize the trading system with unified configuration"""
        # Use the ConfigManager instead of direct YAML loading
        self.config_manager = ConfigManager.get_instance()
        
        # Get configurations from centralized manager
        self.config = self.config_manager.get_config('trading')
        self.model_config = self.config_manager.get_config('model')
        self.strategies_config = self.config_manager.get_config('strategies')
        self.services_config = self.config_manager.get_config('services')
        
        if not self.config:
            raise ValueError("Trading configuration not found")
        
        # Validate required configuration
        self._validate_required_config()
        
        # Initialize empty containers to avoid attributes not found errors
        self.exchange_clients = {}
        
        # Create instances of required services and components
        self.robo_service = RoboService(self.config)
        self.market_data_service = MarketDataService(self.services_config.get('market_data', {}))
        self.news_service = NewsService(self.services_config.get('news', {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.strategies_config.get('sentiment', {}))
        self.market_detector = MarketInefficencyDetector(self.model_config.get('market', {}))
        
        # Initialize portfolio optimizer and risk manager
        self.portfolio_optimizer = PortfolioOptimizer(self.config.get('portfolio', {}))
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        
        # Default setting for model verbosity
        self.model_quiet = False
        
        # Initialize FinGPT model using model config
        # Will be configured with model_quiet during initialize()
        self.fingpt_model = None

    def get_config(self, path: str, default=None):
        """
        Access configuration using dot notation with optional default
        
        Args:
            path: Configuration path in dot notation (e.g. 'trading.pairs')
            default: Default value if path doesn't exist
        """
        parts = path.split('.')
        if not parts:
            raise ValueError(f"Invalid config path: {path}")
            
        # Special case for trading paths
        if parts[0] == 'trading' and len(parts) > 1:
            # FIXED: Only one level of 'trading' needed
            adjusted_path = ['trading'] + parts[1:]
            
            # Start with the config object
            current = self.config
            
            # Navigate through the adjusted path
            for part in adjusted_path:
                if not isinstance(current, dict) or part not in current:
                    if default is not None:
                        return default
                    logger.error(f"Missing required configuration: {path}")
                    raise KeyError(f"Configuration path not found: {path}")
                current = current[part]
                    
            return current
        
        # Handle other config files normally
        elif parts[0] == 'strategies':
            config_obj = self.strategies_config
        elif parts[0] == 'model':
            config_obj = self.model_config
        elif parts[0] == 'services':
            config_obj = self.services_config
        else:
            # Unknown config prefix, use trading as default
            config_obj = self.config
        
        # Standard path traversal
        current = config_obj
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                if default is not None:
                    return default
                logger.error(f"Missing required configuration: {path}")
                raise KeyError(f"Configuration path not found: {path}")
            current = current[part]
            
        return current

    def _validate_required_config(self):
        """Ensure all required configuration parameters exist"""
        required_params = [
            # Core trading parameters
            'trading.pairs',
            'trading.initial_balance',
            'trading.execution.signal_threshold',
            
            # Risk parameters
            'risk.position_limit',
            'risk.max_drawdown',
            
            # Strategy parameters
            'strategies.strategies.sentiment.detection_threshold',
            
            # Other critical parameters
            'trading.loop_interval'
        ]
        
        missing = []
        for param in required_params:
            try:
                self.get_config(param)
            except (KeyError, ValueError) as e:
                missing.append(param)
                
        if missing:
            error_msg = f"Missing required configuration parameters: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def initialize(self, progress_callback=None):
        """Initialize the trading system components with optional progress reporting."""
        logger.info("Starting trading system initialization...")
        
        total_steps = 5
        current_step = 0
        
        try:
            # Initialize empty containers to avoid attributes not found errors
            self.exchange_clients = {}
            
            # Update progress
            if progress_callback:
                progress_callback((current_step / total_steps) * 100)
            current_step += 1
            
            # Set up exchange connections FIRST
            from services.exchanges.binance import BinanceClient
            
            # Initialize exchange clients
            for exchange_config in self.config.get('exchanges', []):
                exchange_name = exchange_config.get('name', '').lower()
                if exchange_name == 'binance':
                    # Get credentials from ConfigManager
                    credentials = self.config_manager.get_exchange_credentials('binance')
                    
                    self.exchange_clients[exchange_name] = await BinanceClient.create({
                        'api_key': credentials['api_key'],
                        'api_secret': credentials['api_secret'],
                        'test_mode': exchange_config.get('test_mode', True),
                        'options': exchange_config.get('options', {})
                    })
                    logger.info(f"Initialized {exchange_name} client (testnet: {exchange_config.get('test_mode', True)})")
            
            # Update progress
            if progress_callback:
                progress_callback((current_step / total_steps) * 100)
            current_step += 1
            
            # NOW initialize services with the created clients
            # Initialize or update market data service with client
            if not hasattr(self, 'market_data_service') or self.market_data_service is None:
                self.market_data_service = MarketDataService(self.services_config.get('market_data', {}))
            await self.market_data_service.setup(self.exchange_clients)
            
            # Initialize or update news service
            if not hasattr(self, 'news_service') or self.news_service is None:
                self.news_service = NewsService(self.services_config.get('news', {}))
            await self.news_service.setup(self.services_config.get('news', {}))
            
            # Initialize FinGPT model with verbosity setting
            self.fingpt_model = FinGPT(self.model_config.get('fingpt', {}))
            
            # Initialize sentiment analyzer with model
            self.sentiment_analyzer = SentimentAnalyzer({
                'model': self.fingpt_model,
                'model_config': self.strategies_config.get('sentiment', {})
            })
            
            # Initialize robo service
            await self.robo_service.setup(
                exchange_clients=self.exchange_clients,
                trading_pairs=self.get_config('trading.pairs'),
                initial_balance=self.get_config('trading.initial_balance')
            )
            
            # Update progress and complete
            if progress_callback:
                progress_callback(100)
            
            logger.info("Trading system initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    async def run(self):
        """Run the main trading system loop."""
        logger.info("Starting trading system main loop...")
        
        try:
            cycle_count = 0
            
            while True:
                cycle_count += 1
                logger.info(f"=== Trading Cycle #{cycle_count} ===")
                
                # Use UI to show cycle information if available
                if hasattr(self, 'ui'):
                    self.ui.console.rule(f"[bold blue]Trading Cycle #{cycle_count}")
                
                try:
                    # Get latest market data
                    await self.update_market_data()
                    
                    # Update UI with latest data
                    await self.update_ui()
                    
                    # Update UI with latest prices if available
                    if hasattr(self, 'ui'):
                        for pair in self.market_data_service.get_watched_pairs():
                            price = self.market_data_service.get_latest_price(pair)
                            if price:
                                self.ui.update_price(pair, price)
                        
                        # Display market data table
                        self.ui.display_market_data()
                    
                    # Check news and sentiment
                    news_data = await self.update_news_data()
                    
                    # Update signals based on sentiment analysis
                    signals = await self.analyze_market_sentiment(news_data)
                    
                    # Process detected signals
                    await self._process_signals(signals)
                    
                    # Add periodic portfolio updates:
                    if cycle_count % 5 == 0:  # Every 5 cycles
                        positions = self.robo_service.get_positions()
                        balance = self.robo_service.get_balance()
                        
                        # Log portfolio information
                        logger.info(f"Current Portfolio: {positions}")
                        logger.info(f"Balance: {balance:.2f} USDT")
                        
                        # Update UI with portfolio information if available
                        if hasattr(self, 'ui'):
                            self.ui.display_portfolio(balance, positions)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    
                    # Display error in UI if available
                    if hasattr(self, 'ui'):
                        self.ui.display_error(f"Trading loop error: {str(e)}")
                
                # Sleep before next cycle
                await asyncio.sleep(self.config.get('cycle_interval', 60))
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {str(e)}")
            # Display fatal error in UI if available
            if hasattr(self, 'ui'):
                self.ui.display_error(f"Fatal trading error: {str(e)}")
            raise

    async def _process_signals(self, signals):
        """Process detected signals and execute trades if needed."""
        try:
            trade_count = 0
            for signal in signals:
                symbol = signal.get('symbol')
                direction = signal.get('direction')
                strength = signal.get('strength', 0.0)
                price = signal.get('price', 0.0)
                
                # Get threshold from config
                threshold = self.get_config('trading.execution_threshold', 0.5)
                
                # Log signal detection
                logger.info(f"Signal detected: {symbol} {direction} (strength: {strength:.2f})")
                
                # Check if signal strength exceeds execution threshold
                if strength >= threshold:
                    # Use rich-compatible symbols
                    logger.info(f"[green]Signal exceeds threshold, executing trade...[/green]")
                    
                    # Execute trade
                    trade_result = await self.robo_service.execute_trade(signal)
                    
                    # Log result
                    if trade_result:
                        trade_count += 1
                        logger.info(f"Trade executed: {trade_result}")
                        
                        # Display in UI if available
                        if hasattr(self, 'ui'):
                            self.ui.display_trade_signal(
                                symbol, 
                                direction, 
                                strength,
                                price,
                                signal.get('metadata', {}).get('confidence', 0.5)
                            )
                else:
                    # Use rich-compatible formatting
                    logger.info(f"[red]Signal below threshold, no trade executed[/red]")
            
            return trade_count
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
            if hasattr(self, 'ui'):
                self.ui.display_error(f"Error processing signals: {str(e)}")
            return 0

    async def execute_trade(self, signal: Dict) -> None:
        """Execute a trade based on a signal"""
        try:
            # Determine position size based on signal strength and risk parameters
            position_size = self._calculate_position_size(signal)
            
            # Format trade for execution
            trade = {
                'symbol': signal['symbol'],
                'type': signal['type'],
                'size': position_size,
                'price': signal['price'],
                'exchange': self.get_config('trading.execution.default_exchange', 'binance')
            }
            
            # Format order size according to exchange requirements
            formatted_size = self._format_order_size(trade)
            trade['size'] = formatted_size
            
            logger.info(f"Executing {trade['type']} trade for {trade['symbol']}: {formatted_size} units")
            
            # Execute on exchange
            client = self.exchange_clients[trade['exchange']]
            if trade['type'].upper() == 'BUY':
                result = await client.create_market_buy_order(trade['symbol'], trade['size'])
            else:
                result = await client.create_market_sell_order(trade['symbol'], trade['size'])
                
            logger.info(f"Trade executed: {result}")
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength and available funds"""
        try:
            # Get account balance
            balance = self.get_config('trading.account.balance', 10000.0)
            max_position_pct = self.get_config('trading.position_sizing.max_position_pct', 0.05)
            base_position_pct = self.get_config('trading.position_sizing.base_position_pct', 0.01)
            
            # Scale position size based on signal strength (0.0-1.0)
            strength_factor = min(signal.get('strength', 0.5), 1.0)
            position_pct = base_position_pct + ((max_position_pct - base_position_pct) * strength_factor)
            
            # Calculate position size in quote currency (e.g., USDT)
            position_size = balance * position_pct

            # Check minimum notional value requirement first
            min_notional = 10.0  # Default minimum value in USDT
            try:
                # Get exchange-specific minimum if available
                if hasattr(self, 'exchange_clients') and 'binance' in self.exchange_clients:
                    min_notional = self.exchange_clients['binance'].get_min_notional(signal['symbol'])
            except Exception as e:
                logger.warning(f"Error getting min notional, using default: {e}")

            # Ensure position meets minimum notional value
            if position_size < min_notional:
                logger.warning(f"Increasing position size from {position_size:.2f} to minimum {min_notional} USDT")
                position_size = min_notional
            
            # Calculate quantity in base currency
            price = signal.get('price', 0.0)
            if price <= 0:
                logger.warning("Invalid price in signal, using fallback price")
                # Try to get current price from market data service
                if hasattr(self, 'market_data_service'):
                    price = self.market_data_service.get_latest_price(signal['symbol'])
                
                if price <= 0:
                    logger.error(f"Cannot calculate position size: no valid price for {signal['symbol']}")
                    return 0.0
            
            quantity = position_size / price
            
            # Get minimum order size from exchange
            symbol = signal['symbol']
            min_notional = 10.0  # Default minimum notional value (e.g., 10 USDT for Binance)
            min_qty = 0.00001    # Default minimum quantity
            
            # Try to get actual minimums from exchange info
            try:
                if hasattr(self, 'exchange_clients') and 'binance' in self.exchange_clients:
                    exchange_info = self.exchange_clients['binance'].get_exchange_info()
                    for symbol_info in exchange_info.get('symbols', []):
                        if symbol_info['symbol'] == symbol:
                            # Extract minimum notional and quantity
                            for filter in symbol_info.get('filters', []):
                                if filter['filterType'] == 'NOTIONAL':
                                    min_notional = float(filter['minNotional'])
                                elif filter['filterType'] == 'LOT_SIZE':
                                    min_qty = float(filter['minQty'])
            except Exception as e:
                logger.warning(f"Error getting exchange minimums: {e}, using defaults")
            
            # Ensure quantity meets minimum requirements
            if quantity < min_qty:
                logger.warning(f"Adjusted quantity from {quantity} to minimum {min_qty}")
                quantity = min_qty
                
            # Ensure notional value meets minimum
            notional = quantity * price
            if notional < min_notional:
                adjusted_qty = min_notional / price
                logger.warning(f"Adjusted quantity to meet min notional: {adjusted_qty}")
                quantity = adjusted_qty
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

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

    async def update_market_data(self):
        """Fetch and process the latest market data"""
        try:
            # Get trading pairs
            pairs = self.get_config('trading.pairs')
            
            # Log request
            logger.debug(f"Fetching market data for pairs: {pairs}")
            
            # Fetch market data
            market_data = await self.market_data_service.get_realtime_quote(pairs)
            
            # Debug log the returned data structure
            logger.debug(f"Market data response: {market_data}")
            
            # Check if data is valid and contains expected fields
            for symbol, data in market_data.items():
                price_fields_found = []
                if isinstance(data, dict):
                    for field in ['price', 'lastPrice', 'last', 'close']:
                        if field in data:
                            price_fields_found.append(f"{field}={data[field]}")
                    logger.debug(f"{symbol} price fields: {', '.join(price_fields_found) or 'none'}")
                else:
                    logger.warning(f"Unexpected data format for {symbol}: {type(data)}")
                
            # Process the market data and update internal state
            # ...
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    async def update_news_data(self):
        """Fetch and process the latest news data"""
        try:
            # Get the trading pairs from config
            pairs = self.get_config('trading.pairs')
            
            # Fetch latest news
            news_data = await self.news_service.fetch_news(pairs)
            
            # Log summary of fetched news
            news_count = len(news_data)
            logger.info(f"Fetched {news_count} news items")
            
            # Process news data by symbol for easier access
            processed_news = {}
            for pair in pairs:
                # Filter news relevant to this pair
                relevant_news = [
                    item for item in news_data 
                    if self.news_service.is_relevant(item, pair)
                ]
                
                processed_news[pair] = relevant_news
                logger.info(f"Fetched {len(relevant_news)} relevant news items for {pair}")
                
            return processed_news
        except Exception as e:
            logger.error(f"Error updating news data: {str(e)}")
            if hasattr(self, 'ui'):
                self.ui.display_error(f"News data error: {str(e)}")
            return {}

    async def analyze_market_sentiment(self, news_data):
        """Analyze market sentiment based on news data"""
        try:
            signals = []
            
            # Return empty list if news_data is None
            if news_data is None:
                logger.warning("No news data available for sentiment analysis")
                return signals
                
            # Get trading pairs and current market prices
            pairs = self.get_config('trading.pairs')
            
            # First get current market prices for all pairs
            current_prices = {}
            for pair in pairs:
                # Get latest price from market data service
                if hasattr(self.market_data_service, 'get_latest_price'):
                    price = self.market_data_service.get_latest_price(pair)
                    current_prices[pair] = price
                else:
                    # Fallback if method doesn't exist
                    current_prices[pair] = 0.0
                        
            for pair in pairs:
                # Skip if no news for this pair
                if pair not in news_data or not news_data[pair]:
                    continue
                    
                # Get current price for this pair
                current_price = current_prices.get(pair, 0.0)
                
                # Skip if price is not available or None
                if current_price is None or current_price <= 0:
                    logger.warning(f"No valid price available for {pair}, skipping sentiment analysis")
                    continue
                    
                # Analyze each news item
                for news in news_data[pair]:
                    try:
                        # Clean the text before analysis
                        text = self._sanitize_text(news.get('title', '') + ' ' + news.get('body', ''))
                        
                        # Check if sentiment_analyzer is properly initialized
                        if not hasattr(self, 'sentiment_analyzer') or self.sentiment_analyzer is None:
                            logger.error("Sentiment analyzer not initialized")
                            continue
                            
                        if not hasattr(self.sentiment_analyzer, 'analyze_text'):
                            logger.error("Sentiment analyzer missing analyze_text method")
                            # Try analyze method instead if available
                            if hasattr(self.sentiment_analyzer, 'analyze'):
                                logger.info("Falling back to analyze method")
                                sentiment_result = await self.sentiment_analyzer.analyze(text)
                            else:
                                logger.error("No sentiment analysis method available")
                                continue
                        else:
                            # Use the correct method
                            sentiment_result = await self.sentiment_analyzer.analyze_text(text)
                        
                        # Check if sentiment_result is None
                        if not sentiment_result:
                            logger.warning("Sentiment analysis returned None result")
                            continue
                            
                        sentiment_score = sentiment_result.get('sentiment', 0.0)
                        confidence = sentiment_result.get('confidence', 0.0)
                        
                        # Log the sentiment analysis using rich-compatible formatting
                        if sentiment_score > 0:
                            logger.info(f"Sentiment analysis: score=[green]{sentiment_score:.2f}[/green], confidence={confidence:.2f}")
                        elif sentiment_score < 0:
                            logger.info(f"Sentiment analysis: score=[red]{sentiment_score:.2f}[/red], confidence={confidence:.2f}")
                        else:
                            logger.info(f"Sentiment analysis: score=[yellow]{sentiment_score:.2f}[/yellow], confidence={confidence:.2f}")
                        
                        # Generate signal if sentiment is strong enough
                        threshold = self.get_config('strategies.sentiment.detection_threshold', 0.3)
                        if abs(sentiment_score) > threshold and confidence > 0.4:
                            logger.info(f"[bold]Strong sentiment signal detected![/bold] (threshold={threshold:.2f})")
                            
                            signals.append({
                                'symbol': pair,
                                'type': 'SENTIMENT',
                                'direction': 'BUY' if sentiment_score > 0 else 'SELL',
                                'strength': abs(sentiment_score) * confidence,
                                'price': current_price,  # Use current market price instead of 0.0
                                'timestamp': datetime.now(),
                                'metadata': {
                                    'sentiment': sentiment_score,
                                    'confidence': confidence,
                                    'news_id': news.get('id'),
                                    'source': news.get('source')
                                }
                            })
                        else:
                            logger.info(f"Sentiment below thresholds, no signal generated")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing news sentiment: {str(e)}")
            
            return signals
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {str(e)}")
            return []

    async def update_ui(self):
        """Update the UI with latest market data"""
        try:
            # Get trading pairs
            pairs = self.get_config('trading.pairs')
            
            for pair in pairs:
                # Fetch price directly from cache where we know it's stored correctly
                price = 0.0
                if hasattr(self, 'market_data_service') and pair in self.market_data_service.cache:
                    data = self.market_data_service.cache[pair]
                    
                    # Try different price fields
                    for field in ['price', 'lastPrice', 'last', 'close']:
                        if field in data and data[field]:
                            try:
                                price = float(data[field])
                                if price > 0:
                                    # Update UI with the valid price
                                    if hasattr(self, 'ui'):
                                        self.ui.update_price(pair, price)
                                    break
                            except (ValueError, TypeError):
                                pass
                
                # Calculate 24h change if we have enough history data
                change_pct = 0.0
                if hasattr(self, 'market_data_service') and \
                   hasattr(self.market_data_service, 'price_history') and \
                   pair in self.market_data_service.price_history:
                    
                    history = self.market_data_service.price_history[pair]
                    if len(history) >= 2:
                        # Calculate change from oldest to newest price
                        newest_price = history[-1]['price']
                        # Find price closest to 24h ago
                        day_ago = datetime.now() - timedelta(hours=24)
                        closest_idx = 0
                        for i, entry in enumerate(history):
                            if abs((entry['timestamp'] - day_ago).total_seconds()) < \
                               abs((history[closest_idx]['timestamp'] - day_ago).total_seconds()):
                                closest_idx = i
                        
                        if closest_idx < len(history):
                            old_price = history[closest_idx]['price']
                            if old_price > 0:
                                change_pct = ((newest_price - old_price) / old_price) * 100
                                if hasattr(self, 'ui'):
                                    self.ui.update_change(pair, change_pct)
            
            # Update sentiment in UI
            # ...
        except Exception as e:
            logger.error(f"Error updating UI: {str(e)}")

# Helper function to wait for shutdown signal
async def wait_for_shutdown(shutdown_event):
    await shutdown_event.wait()
    logger.info("Shutdown event triggered")

def configure_logging(args):
    """Configure logging based on verbosity level arguments."""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Determine appropriate logging level
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Use Rich-compatible logging if not in silent mode
    if not args.silent:
        from utils.logging import LogManager
        log_manager = LogManager({
            "log_dir": "logs",
            "log_file": args.log_file if hasattr(args, 'log_file') else None
        })
        log_manager.setup_rich_logging(level)
    else:
        # Silent mode - only log to file, not console
        # Get the root logger and remove any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set up file logging if requested
        if hasattr(args, 'log_file') and args.log_file:
            file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.DEBUG)
    
    return level

def parse_arguments():
    """Parse command line arguments for the FinGPT Trader application."""
    parser = argparse.ArgumentParser(description='FinGPT Trader - AI-powered trading system')
    
    # Verbosity control arguments (mutually exclusive)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument('--quiet', '-q', action='store_true', 
                    help='Show only essential information (trades, major events)')
    verbosity_group.add_argument('--verbose', '-v', action='store_true', 
                    help='Show full debugging output')
    
    # Log file option
    parser.add_argument('--log-file', '-l', type=str, metavar='FILE',
                    default='logs/fingpt_trader.log',
                    help='Write detailed logs to specified file')
    
    # Add model output suppression (can be used with any verbosity level)
    parser.add_argument('--model-quiet', '-mq', action='store_true',
                    help='Suppress model technical output')
    
    parser.add_argument('--silent', '-s', action='store_true',
                help='Silent mode - suppress all output (equivalent to -q -mq)')
    
    return parser.parse_args()

async def main():
    """Main entry point for the FinGPT Trader application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Import verbosity manager here to avoid circular imports
    from utils.verbosity import VerbosityManager
    vm = VerbosityManager.get_instance()
    
    # Set verbosity level
    if args.quiet:
        vm.set_level(VerbosityManager.QUIET)
    elif args.verbose:
        vm.set_level(VerbosityManager.VERBOSE)
    else:
        vm.set_level(VerbosityManager.NORMAL)
    
    # Control model output separately if requested
    vm.set_suppress_model_output(args.model_quiet)
    
    # Add handling for the --silent flag
    if args.silent:
        vm.silence_all()

    # Configure logging based on verbosity arguments
    logging_level = configure_logging(args)
    
    # Initialize rich console UI if not in silent mode
    if not args.silent and not args.quiet:
        from utils.console_ui import ConsoleUI
        ui = ConsoleUI.get_instance()
        ui.setup(watched_pairs=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        
        # Set verbosity level for UI too
        ui.set_verbose(args.verbose)

    # Log startup information
    logging.info("Starting FinGPT Trader")
    logging.debug(f"Verbosity level: {logging.getLevelName(logging_level)}")
    
    # Continue with existing initialization code
    system = None
    try:
        # Create and initialize using ConfigManager
        system = TradingSystem()
        
        # Pass verbosity information to TradingSystem
        system.verbosity_manager = vm
        
        # Add UI instance to trading system if available
        if not args.silent and not args.quiet:
            system.ui = ui
            
            # Show progress bar for model loading
            with ui.create_progress_bar("Loading trading model")[0] as progress:
                # Update progress as initialization proceeds
                await system.initialize(progress_callback=lambda p: progress.update(task_id=0, completed=p))
        else:
            await system.initialize()
            
        # Run the main system loop
        await system.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        if not args.silent and not args.quiet and 'ui' in locals():
            ui.display_error(f"Fatal error: {str(e)}")
    finally:
        # Ensure proper cleanup even if there's an error
        if system:
            await system.shutdown()
        
        # Clean up verbosity manager
        vm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
