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
        
        # Initialize FinGPT model with verbosity setting
        self.fingpt_model = FinGPT(self.model_config.get('fingpt', {}))

        # Initialize sentiment analyzer with model
        self.sentiment_analyzer = SentimentAnalyzer({
            'model': self.fingpt_model,  # Pass model instance explicitly
            'model_config': self.strategies_config.get('sentiment', {})
        })

        # Initialize system monitoring
        from services.monitoring.system_monitor import SystemMonitor
        self.system_monitor = SystemMonitor(self.config.get('monitoring', {}))
        
        # Initialize performance tracking
        from services.monitoring.performance_tracker import PerformanceTracker
        self.performance_tracker = PerformanceTracker(self.config.get('performance', {}))

        # Add these default attributes
        self.price_history = {}  # For storing historical prices
        self.trade_history = []  # For tracking trade attempts
        self.recent_signals = [] # For storing recent trading signals

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
                try:
                    cycle_count += 1
                    logger.info(f"=== Trading Cycle #{cycle_count} ===")
                    
                    # MONITOR SYSTEM RESOURCES
                    if hasattr(self, 'system_monitor'):
                        system_metrics = await self.system_monitor.check_health()
                        if system_metrics.get('status') == 'critical':
                            logger.warning(f"System resources critical: {system_metrics.get('critical_metrics')}")
                            if hasattr(self, 'ui'):
                                self.ui.display_warning(f"System resources low: {system_metrics.get('critical_metrics')}")
                    
                    # Display UI header with performance metrics if available
                    if hasattr(self, 'performance_tracker'):
                        try:
                            performance_metrics = await self.performance_tracker.calculate_metrics()
                            
                            # Pass metrics to UI header
                            if hasattr(self, 'ui') and hasattr(self.ui, 'display_trading_cycle_header'):
                                self.ui.display_trading_cycle_header(cycle_count, performance_metrics)
                                self.ui.performance_metrics = performance_metrics
                        except Exception as e:
                            logger.error(f"Error calculating performance metrics: {e}")
                            if hasattr(self, 'ui') and hasattr(self.ui, 'display_trading_cycle_header'):
                                self.ui.display_trading_cycle_header(cycle_count)
                    else:
                        # Just display basic header if no metrics available
                        if hasattr(self, 'ui') and hasattr(self.ui, 'display_trading_cycle_header'):
                            self.ui.display_trading_cycle_header(cycle_count)
                    
                    # STEP 1: UPDATE MARKET DATA - Add this critical step
                    await self.update_market_data()
                    
                    # STEP 2: UPDATE NEWS DATA - Get latest news
                    news_data = await self.update_news_data()
                    
                    # STEP 3: UPDATE PORTFOLIO AND UI
                    await self.update_ui()
                    
                    # STEP 4: ANALYZE SENTIMENT AND GENERATE SIGNALS
                    if news_data:
                        signals = await self.analyze_market_sentiment(news_data)
                        
                        # STEP 5: PROCESS SIGNALS AND EXECUTE TRADES
                        if signals:
                            trade_count = await self._process_signals(signals)
                            logger.info(f"Processed {len(signals)} signals, executed {trade_count} trades")
                    
                    # PERFORMANCE TRACKING - Update with portfolio data
                    if hasattr(self, 'performance_tracker') and cycle_count % 5 == 0:
                        # Get portfolio data from RoboService
                        portfolio_summary = self.robo_service.get_portfolio_summary()
                        
                        # Update performance tracker
                        await self.performance_tracker.record_performance(portfolio_summary)
                        
                        # Calculate and display key metrics every 10 cycles
                        if cycle_count % 10 == 0:
                            try:
                                performance_metrics = await self.performance_tracker.calculate_metrics()
                                
                                # Display forex-style performance metrics
                                logger.info(f"===== PERFORMANCE METRICS =====")
                                logger.info(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 'N/A')}")
                                logger.info(f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
                                logger.info(f"Profit Factor: {performance_metrics.get('profit_factor', 'N/A')}")
                                logger.info(f"Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
                                
                                # Display in UI if available
                                if hasattr(self, 'ui') and hasattr(self.ui, 'display_performance_metrics'):
                                    self.ui.display_performance_metrics(performance_metrics)
                                    self.ui.performance_metrics = performance_metrics
                            except Exception as e:
                                logger.error(f"Error displaying performance metrics: {e}")
                    
                    # After all data is updated
                    if hasattr(self, 'ui'):
                        # Store cycle number
                        self.ui.current_cycle = cycle_count
                        
                        # Display the complete dashboard instead of individual sections
                        if hasattr(self.ui, 'display_trader_dashboard'):
                            self.ui.display_trader_dashboard()
                        else:
                            # Fallback to older update method
                            await self.update_ui()
                    
                    # Sleep before next cycle
                    await asyncio.sleep(self.config.get('trading.loop_interval', 60))
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    
                    # Display error in UI if available
                    if hasattr(self, 'ui'):
                        self.ui.display_error(f"Trading loop error: {str(e)}")
            
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
            # CHECK CIRCUIT BREAKERS FIRST
            if hasattr(self, 'system_monitor') and hasattr(self.system_monitor, 'circuit_breaker'):
                # Get market data for circuit breaker check
                market_data = {}
                if hasattr(self, 'market_data_service'):
                    for pair in self.market_data_service.get_watched_pairs():
                        market_data[pair] = self.market_data_service.get_market_data(pair)
                
                # Check if circuit breaker is triggered
                if self.system_monitor.circuit_breaker.check_conditions(market_data):
                    logger.warning("[red]Circuit breaker triggered! All trading halted.[/red]")
                    if hasattr(self, 'ui'):
                        self.ui.display_warning("⚠️ CIRCUIT BREAKER TRIGGERED - Trading halted")
                    return 0  # No trades executed
                    
            # Check for rapid price moves
            if hasattr(self, 'system_monitor') and hasattr(self.system_monitor, 'detect_rapid_moves'):
                rapid_moves = self.system_monitor.detect_rapid_moves(market_data)
                if rapid_moves:
                    affected_pairs = ", ".join(rapid_moves.keys())
                    logger.warning(f"[yellow]Rapid price moves detected in: {affected_pairs}[/yellow]")
                    if hasattr(self, 'ui'):
                        self.ui.display_warning(f"⚠️ Price volatility detected in {affected_pairs}")
            
            # Continue with regular signal processing
            trade_count = 0
            for signal in signals:
                symbol = signal.get('symbol')
                direction = signal.get('direction')
                strength = signal.get('strength', 0.0)
                price = signal.get('price', 0.0)
                
                # Boost signal strength when multiple indicators align
                if any(s['symbol'] == symbol and s != signal for s in signals):
                    # Multiple signals for same asset - confirmation boost
                    logger.info(f"Multiple signals detected for {symbol}, boosting strength")
                    strength = min(strength * 1.25, 1.0)  # Boost by 25% but cap at 1.0
                    signal['strength'] = strength
                    
                    # Multiple signals for same asset - dynamic threshold adjustment
                    logger.info(f"Multiple signals detected for {symbol}, adjusting threshold")
                    # More aggressive threshold when multiple signals confirm the same direction
                    dynamic_threshold = self.get_config('trading.execution.signal_threshold', 0.5) * 0.8
                    
                    if strength >= dynamic_threshold:
                        logger.info(f"Signal exceeds dynamic threshold ({dynamic_threshold:.2f}), executing trade...")
                        trade_result = await self.robo_service.execute_trade(signal)

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

    async def process_trading_signal(self, signal):
        """Process and execute a trading signal with enhanced validation"""
        if not signal or 'symbol' not in signal:
            return
            
        # Validate strength exceeds threshold
        strength = signal.get('strength', 0)
        threshold = self.get_config('trading.execution.signal_threshold', 0.5)
        
        if strength >= threshold:
            logger.info(f"Signal exceeds threshold, executing trade...")
            
            # Execute the trade with our trading service
            side = signal.get('side', 'BUY')
            result = await self.robo_service.execute_trade(signal)
            
            # Record trade in performance tracker
            if hasattr(self, 'performance_tracker'):
                if result.get('success', False):
                    # Successful trade
                    await self.performance_tracker.record_trade({
                        'symbol': signal.get('symbol'),
                        'side': side,
                        'entry_price': result.get('price', 0),
                        'quantity': result.get('quantity', 0),
                        'timestamp': datetime.now(),
                        'signal_strength': strength,
                        'signal_type': signal.get('type', 'sentiment')
                    })
                elif result.get('action') == 'ACCUMULATING':
                    # Accumulating trade
                    await self.performance_tracker.record_pending_order({
                        'symbol': signal.get('symbol'),
                        'side': side,
                        'accumulated': result.get('accumulated', 0),
                        'value': result.get('accumulated_value', 0),
                        'timestamp': datetime.now()
                    })
            
            return result
        else:
            logger.info(f"Signal below threshold ({threshold:.2f}), no trade executed")
            return None

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
            
    async def handle_minimum_size_error(self, signal: Dict, error_msg: str):
        """
        Handle cases where position size is too small.
        
        Args:
            signal: The trading signal that couldn't be executed
            error_msg: The error message
        """
        try:
            # Log the issue with clear formatting
            logger.warning("═════════════════════════════════════════════════════")
            logger.warning(f"POSITION SIZING ERROR: {signal['symbol']}")
            logger.warning(f"Signal strength: {signal.get('strength', 0):.2f}")
            logger.warning(f"Error: {error_msg}")
            
            # Store in trade history for reporting
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': signal['symbol'],
                'side': signal.get('side', 'UNKNOWN'),
                'strength': signal.get('strength', 0),
                'status': 'FAILED',
                'reason': 'MINIMUM_SIZE',
                'price': signal.get('price', 0)
            })
            
            # Create a suggested allocation based on current price
            price = signal.get('price', 0)
            if price > 0:
                # For BTC, suggest size in USDT that would meet minimums
                min_qty = 1e-5 if signal['symbol'].startswith('BTC') else 1e-4
                suggested_notional = min_qty * price * 1.1  # Add 10% buffer
                logger.warning(f"Suggested minimum allocation: {suggested_notional:.2f} USDT")
                
                # Suggest portfolio percentage 
                balance = self.get_config('trading.account.balance', 10000.0)
                suggested_pct = (suggested_notional / balance) * 100
                logger.warning(f"Suggested min position size: {suggested_pct:.2f}% of portfolio")
                
            logger.warning("═════════════════════════════════════════════════════")
        except Exception as e:
            logger.error(f"Error in handling minimum size notification: {e}")

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength and portfolio value"""
        try:
            # Get trading pair symbol and signal strength
            symbol = signal.get('symbol', '')
            strength = signal.get('strength', 0.5)
            confidence = signal.get('confidence', 0.5)
            
            # Get portfolio value
            portfolio_value = 0
            if hasattr(self, 'robo_service') and hasattr(self.robo_service, 'portfolio'):
                portfolio = self.robo_service.portfolio
                portfolio_value = portfolio.total_value()
            
            # Default to small position if we can't get portfolio value
            if portfolio_value <= 0:
                portfolio_value = 10000  # Assume default $10K
            
            # Get current price
            price = signal.get('price', 0)
            if price <= 0 and hasattr(self, 'market_data_service'):
                price = self.market_data_service.get_latest_price(symbol)
            
            if price <= 0:
                logger.error(f"Cannot calculate position size: no price for {symbol}")
                return 0
                
            # Get position sizing settings
            base_position = self.get_config('trading.position_sizing.base', 0.01)  # Default 1%
            max_position = self.get_config('trading.position_sizing.max', 0.1)    # Default 10%
            
            # Calculate scaled position size based on strength and confidence
            confidence_factor = confidence if confidence > 0.3 else 0.3
            combined_signal = strength * confidence_factor
            
            # Scale position between base and max
            position_pct = base_position + (combined_signal * (max_position - base_position))
            
            # Check for momentum
            prices = []
            if hasattr(self, 'market_data_service'):
                prices = self.market_data_service.get_recent_prices(symbol, limit=3)
            
            # Adjust position size based on momentum
            if len(prices) >= 3:
                if prices[-1] > prices[-2] > prices[-3] and signal.get('side') == 'BUY':
                    logger.info(f"Upward momentum detected for {symbol}, increasing position size")
                    position_pct *= 1.2  # 20% larger position on momentum
                elif prices[-1] < prices[-2] < prices[-3] and signal.get('side') == 'SELL':
                    logger.info(f"Downward momentum detected for {symbol}, increasing position size")
                    position_pct *= 1.2  # 20% larger position on momentum
            
            # Calculate position value
            position_value = portfolio_value * position_pct
            
            # Convert to quantity based on current price
            quantity = position_value / price
            
            # Fix: Get exchange minimum requirements
            min_qty = 0.0001  # Default minimum quantity
            min_notional = 15.0  # Default minimum notional value ($15)
            
            # Get exchange-specific requirements if robo_service is available
            if hasattr(self, 'robo_service') and hasattr(self.robo_service, 'get_exchange_minimum_requirements'):
                # Since we're in a non-async context, we need to use a synchronous alternative
                # or call this earlier in an async context
                symbol_info = self.get_config(f'exchange.minimum_requirements.{symbol}', {})
                min_qty = symbol_info.get('min_qty', min_qty)
                min_notional = symbol_info.get('min_notional', min_notional)
            
            # Fix: Ensure position meets minimum requirements
            if quantity * price < min_notional:
                # Adjust to meet minimum notional value
                logger.info(f"Adjusting position size to meet minimum notional of ${min_notional}")
                quantity = (min_notional * 1.01) / price  # Add 1% buffer
            
            if quantity < min_qty:
                # Adjust to meet minimum quantity
                logger.info(f"Adjusting position size to meet minimum quantity of {min_qty}")
                quantity = min_qty * 1.01  # Add 1% buffer
            
            return quantity
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

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
        """Fetch and process the latest market data with enhanced reliability"""
        try:
            # Get trading pairs
            pairs = self.get_config('trading.pairs')
            
            # Fetch market data with primary method
            market_data = await self.market_data_service.get_realtime_quote(pairs)
            
            # If no data received, try direct Binance API fallback
            if not market_data or len(market_data) < len(pairs):
                missing_pairs = [p for p in pairs if p not in market_data]
                if missing_pairs:
                    logger.warning(f"Missing data for {', '.join(missing_pairs)}, using direct API fallback")
                    direct_data = await self.market_data_service.fetch_prices_directly(missing_pairs)
                    
                    # Merge results
                    if direct_data:
                        market_data.update(direct_data)
            
            # Ensure market_data is properly synced
            if hasattr(self.market_data_service, '_sync_market_data_from_cache'):
                self.market_data_service._sync_market_data_from_cache()
            
            # Log success/failure with pricing information
            if market_data:
                prices_str = ", ".join([f"{s}: {p:.2f}" for s, p in market_data.items()])
                logger.info(f"Fetched prices: {prices_str}")
            else:
                logger.warning("Failed to fetch any market data after all attempts")
                return
            
            # Update UI and portfolio with the new prices
            for symbol, price in market_data.items():
                # Update UI with new price
                if hasattr(self, 'ui'):
                    self.ui.update_price(symbol, price)
                
                # Update portfolio prices
                if hasattr(self, 'robo_service') and hasattr(self.robo_service, 'portfolio'):
                    if not hasattr(self.robo_service.portfolio, 'last_prices'):
                        self.robo_service.portfolio.last_prices = {}
                    self.robo_service.portfolio.last_prices[symbol] = price
            
            # Calculate and update 24h changes
            if hasattr(self.market_data_service, 'calculate_price_changes'):
                changes = self.market_data_service.calculate_price_changes()
                
                # Update UI with price changes
                if hasattr(self, 'ui'):
                    for symbol, change in changes.items():
                        self.ui.update_change(symbol, change)
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

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
            
            # Get sentiment thresholds from config
            detection_threshold = self.get_config('strategies.sentiment.detection_threshold', 0.3)
            confidence_threshold = self.get_config('strategies.sentiment.min_confidence', 0.4)
            
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
                        
                        # Generate signal if sentiment is strong enough - using config thresholds
                        if abs(sentiment_score) > detection_threshold and confidence > confidence_threshold:
                            logger.info(f"[bold]Strong sentiment signal detected![/bold] (threshold={detection_threshold:.2f})")
                            
                            signals.append({
                                'symbol': pair,
                                'type': 'SENTIMENT',
                                'direction': 'BUY' if sentiment_score > 0 else 'SELL',
                                'strength': abs(sentiment_score) * confidence,
                                'price': current_price,
                                'timestamp': datetime.now(),
                                'metadata': {
                                    'sentiment': sentiment_score,
                                    'confidence': confidence,
                                    'news_id': news.get('id'),
                                    'source': news.get('source')
                                }
                            })
                            
                            # Update UI with new sentiment
                            if hasattr(self, 'ui'):
                                sentiment_text = self.ui._format_sentiment(sentiment_score)
                                self.ui.update_sentiment(pair, sentiment_text)
                        else:
                            logger.info(f"Sentiment below thresholds, no signal generated")
                        
                        # After sentiment analysis, add:
                        if hasattr(self, 'ui'):
                            sentiment_display = self.ui._format_sentiment(sentiment_score)
                            self.ui.update_sentiment(pair, sentiment_display)
                        
                        # Update in market data service
                        if hasattr(self, 'market_data_service'):
                            # Store sentiment in market data for display
                            market_data = self.market_data_service.get_latest_data(pair)
                            if isinstance(market_data, dict):
                                market_data['sentiment'] = sentiment_score
                                market_data['sentiment_confidence'] = confidence
                            
                            # Update market data
                            self.market_data_service.update_symbol_data(pair, market_data)
                            
                    except Exception as e:
                        logger.error(f"Error analyzing news sentiment: {str(e)}")
            
            return signals
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {str(e)}")
            return []

    async def update_ui(self):
        """Update the UI with latest market data"""
        try:
            # Update portfolio positions with CURRENT PRICES
            if hasattr(self, 'robo_service') and hasattr(self.robo_service, 'portfolio'):
                portfolio = self.robo_service.portfolio
                positions = portfolio.positions  # Dictionary of positions
                balance = portfolio.cash
                
                # Calculate total value with current prices
                total_value = balance
                
                # Get entry prices from portfolio
                entry_prices = {}
                if hasattr(self.robo_service.portfolio, 'position_entries'):
                    entry_prices = self.robo_service.portfolio.position_entries

                # Make sure to log entry prices for debugging
                logger.debug(f"Entry prices: {entry_prices}")

                # Get current prices for each position
                current_prices = {}
                for symbol in positions.keys():
                    # Get current price from market data service
                    price = 0.0
                    if hasattr(self.market_data_service, 'get_latest_price'):
                        price = self.market_data_service.get_latest_price(symbol)
                        current_prices[symbol] = price
                        
                        # Update last_prices in portfolio to ensure proper valuation
                        if hasattr(portfolio, 'last_prices'):
                            portfolio.last_prices[symbol] = price
                            
                        # Calculate position value (important!)
                        position_size = positions.get(symbol, 0)
                        position_value = position_size * price
                        total_value += position_value
                
                # Display portfolio with entry prices and current prices
                if hasattr(self, 'ui'):
                    self.ui.display_portfolio(balance, positions, entry_prices, current_prices)
                    self.ui.portfolio_cash = balance
                    self.ui.portfolio_positions = positions
                    self.ui.position_entries = entry_prices
                    self.ui.current_prices = current_prices
            
            # Update market data UI component if available
            if hasattr(self, 'ui') and hasattr(self.market_data_service, 'get_latest_data'):
                market_data = {}
                pairs = self.get_config('trading.pairs')
                
                for pair in pairs:
                    data = self.market_data_service.get_latest_data(pair)
                    if data:
                        market_data[pair] = data
                
                if market_data and hasattr(self.ui, 'display_market_data'):
                    self.ui.display_market_data(market_data)
                    self.ui.market_data = market_data
            
            # Update trading signals UI if there are any recent signals
            if hasattr(self, 'recent_signals') and self.recent_signals and hasattr(self.ui, 'display_signals'):
                self.ui.display_signals(self.recent_signals)
            
            # Update system health indicator
            if hasattr(self, 'system_monitor') and hasattr(self.ui, 'update_system_health'):
                health = await self.system_monitor.check_health()
                self.ui.update_system_health(health)
        
        except Exception as e:
            logger.error(f"Error updating UI: {str(e)}")

if __name__ == "__main__":
    import asyncio
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="FinGPT Trader")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimize console output")
    parser.add_argument("-mq", "--model-quiet", action="store_true", help="Suppress model initialization output")
    args = parser.parse_args()
    
    # Create and run the trading system
    trading_system = TradingSystem()
    
    # Set verbosity based on arguments
    if args.verbose:
        trading_system.model_quiet = False
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        trading_system.model_quiet = True
        logger.setLevel(logging.WARNING)
    elif args.model_quiet:
        trading_system.model_quiet = True
    
    # Create console UI
    from utils.console_ui import ConsoleUI
    trading_system.ui = ConsoleUI.get_instance()
    trading_system.ui.set_verbose(args.verbose)

    # Initialize UI with trading pairs from config
    pairs = trading_system.get_config('trading.pairs', [])
    trading_system.ui.setup(watched_pairs=pairs, display_header=True)
    
    # Run the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize and run the trading system
        loop.run_until_complete(trading_system.initialize())
        loop.run_until_complete(trading_system.run())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        # Cleanup resources
        loop.run_until_complete(trading_system.shutdown())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        trading_system.ui.display_error(f"Fatal error: {str(e)}")
    finally:
        loop.close()