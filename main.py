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
    "level": "INFO"
}).setup_basic_logging()
logger = logging.getLogger(__name__)

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.market.inefficiency import MarketInefficencyDetector
from strategies.sentiment.analyzer import SentimentAnalyzer  # Change import to use correct analyzer
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
        update_portfolio(): Update portfolio state
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
        load_dotenv(override=True, verbose=True)  # Add verbose=True to debug env loading
        
        # Enhanced environment variable logging
        api_key = os.environ.get('BINANCE_API_KEY')
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        logger.info(f"API Key present: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
        logger.info(f"API Secret present: {bool(api_secret)} (length: {len(api_secret) if api_secret else 0})")
        
        # Verify required environment variables
        required_env = {
            'BINANCE_API_KEY': 'Binance API key',
            'BINANCE_API_SECRET': 'Binance API secret'  # Changed from BINANCE_SECRET_KEY
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
            
        # Remove env var processing since we'll use direct environment access
        # self._process_env_vars(self.config)  # Comment out or remove this line
        
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
            'model': self.fingpt_model,  # Pass model instance
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

    def _process_env_vars(self, config: dict) -> None:
        """Replace ${VAR} with environment variable values"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    if env_value is None:
                        raise ValueError(f"Environment variable {env_var} not found")
                    logger.debug(f"Replacing {env_var} with value of length {len(env_value)}")
                    config[key] = str(env_value)  # Ensure string type
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, (dict, list)):
                    self._process_env_vars(item)

    async def initialize(self):
        """
        Initialize all system components sequentially.
        
        Initialization Order:
        1. Sentiment Analysis Engine
            - Load ML models
            - Initialize news feeds
        2. Market Detection System
            - Configure technical indicators
            - Load historical data
        3. Exchange Connections
            - Setup API clients
            - Verify connectivity
        4. Market State
            - Initialize order books
            - Setup data streams
        5. Robo-Advisory Service
            - Load client profiles
            - Initialize portfolio models
            
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            logger.info("Initializing trading system...")
            
            # Initialize data services first
            await self.news_service._setup()
            await self.market_data_service.start()
            logger.info("Data services initialized")
            
            # 1. Initialize base components
            await self.sentiment_analyzer.initialize()
            await self.market_detector.initialize()
            
            # 2. Setup exchange connections
            for exchange_config in self.config.get('exchanges', []):
                client = await self._setup_exchange_client(exchange_config)
                self.exchange_clients[exchange_config['name']] = client
            
            # 3. Initialize market state
            self.market_state = await self._initialize_market_state()
            
            # 4. Initialize robo service
            await self.robo_service.initialize()
            
            self.is_running = True
            logger.info("Trading system initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.shutdown()
            raise

    async def shutdown(self):
        """Cleanup system resources"""
        logger.info("\nShutting down trading system...")
        try:
            # First stop all active services
            self.is_running = False
            
            # Cleanup data services first to stop data flow
            if hasattr(self, 'market_data_service'):
                await self.market_data_service.stop()
            if hasattr(self, 'news_service'):
                await self.news_service._cleanup()
                
            # Cleanup analysis components
            if hasattr(self, 'sentiment_analyzer'):
                await self.sentiment_analyzer.cleanup()
            if hasattr(self, 'market_detector'):
                await self.market_detector.cleanup()
            if hasattr(self, 'robo_service'):
                await self.robo_service.cleanup()
            
            # Close exchange connections last
            for client in self.exchange_clients.values():
                await client.close()
            
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
        finally:
            # Force exit after cleanup
            sys.exit(0)

    async def _setup_exchange_client(self, exchange_config: Dict):
        """Setup exchange client connection"""
        try:
            exchange_type = exchange_config.get('name', '').lower()
            
            if exchange_type == 'binance':
                from services.exchanges.binance import BinanceClient
                
                # Get credentials directly from environment
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_API_SECRET')
                
                if not api_key or not api_secret:
                    logger.error("Missing Binance credentials in environment")
                    raise ValueError("Binance API credentials not set")

                # Create client config dict
                client_config = {
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'test_mode': exchange_config.get('test_mode', True)
                }
                
                client = await BinanceClient.create(client_config)
                logger.info(f"Successfully connected to {exchange_type}")
                return client
                
        except Exception as e:
            logger.error(f"Failed to setup {exchange_type} client: {str(e)}")
            raise

    async def _initialize_market_state(self) -> Dict:
        """Initialize market state with required data"""
        state = {}
        for exchange, client in self.exchange_clients.items():
            # Get configured trading pairs instead of all pairs
            configured_pairs = self.config.get('trading', {}).get('pairs', [])
            logger.info(f"Initializing {len(configured_pairs)} configured pairs for {exchange}")
            
            # Get initial market data
            state[exchange] = {
                'pairs': configured_pairs,
                'orderbooks': {},
                'trades': {},
                'candles': {}
            }
            
            # Initialize data for each configured pair
            for pair in configured_pairs:
                logger.info(f"Loading market data for {pair}...")
                state[exchange]['orderbooks'][pair] = await client.get_orderbook(pair)
                state[exchange]['trades'][pair] = await client.get_recent_trades(pair)
                state[exchange]['candles'][pair] = await client.get_candles(pair)
                
                # Log current price
                if state[exchange]['trades'][pair]:
                    latest_price = float(state[exchange]['trades'][pair][-1]['price'])
                    logger.info(f"Current {pair} price: {latest_price:.2f}")
                    
                # Add status indicators
                state[exchange]['status'] = {
                    'active': True,
                    'timestamp': datetime.now().timestamp(),
                    'initialized_pairs': configured_pairs
                }
                
        return state

    async def get_market_data(self, symbol: Optional[str] = None) -> Dict:
        """Get market data for analysis"""
        if symbol and not await self._is_valid_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")

        data = {}
        for exchange, client in self.exchange_clients.items():
            if symbol:
                pairs = [symbol]
            else:
                pairs = self.market_state[exchange]['pairs']

            exchange_data = {}
            for pair in pairs:
                # Get ticker data first
                ticker = await client.get_ticker(pair)
                
                exchange_data[pair] = {
                    'orderbook': await client.get_orderbook(pair),
                    'trades': await client.get_recent_trades(pair),
                    'candles': await client.get_candles(pair),
                    'volume': float(ticker.get('volume', 0))  # Get volume from ticker
                }
            data[exchange] = exchange_data

        return data

    async def detect_inefficiencies(self, market_data: Dict) -> Dict:
        """Analyze market data for trading opportunities."""
        signals = {}
        for exchange, exchange_data in market_data.items():
            for pair, pair_data in exchange_data.items():
                # Process market data
                processed_data = self._preprocess_market_data(pair_data)
                
                # Get sentiment analysis and convert to time series
                news_data = await self._fetch_relevant_news(pair)
                combined_text = " ".join(news_data)
                # Add await here
                sentiment_result = await self.sentiment_analyzer.analyze(combined_text)
                
                # Create sentiment Series with same index as price data
                sentiment_series = pd.Series(
                    [sentiment_result['compound']] * len(processed_data['prices']),
                    index=processed_data['prices'].index
                )
                
                # Detect inefficiencies with time-indexed sentiment
                signal = self.market_detector.detect_inefficiencies(
                    prices=processed_data['prices'],
                    volume=processed_data['volume'],
                    sentiment=sentiment_series
                )
                
                signals[f"{exchange}_{pair}"] = signal
        
        return signals

    def _preprocess_market_data(self, pair_data: Dict) -> Dict:
        """Preprocess market data for analysis"""
        candles = np.array(pair_data['candles'])
        
        # Create DataFrame with all columns
        df = pd.DataFrame(candles, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_base_vol',
            'taker_quote_vol', 'ignore'
        ])
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create index using timestamp
        df.set_index('open_time', inplace=True)
        
        return {
            'prices': df,  # DataFrame with time index
            'volume': pd.Series(df['volume'].values, index=df.index),  # Series for rolling calculations
            'timestamp': df.index.values
        }

    async def _fetch_relevant_news(self, pair: str) -> List[str]:
        """Fetch relevant news for the trading pair"""
        try:
            # Convert pair to searchable terms (e.g., BTCUSDT -> "Bitcoin cryptocurrency")
            base_asset = pair.replace('USDT', '').replace('USD', '')
            search_terms = {
                'BTC': 'Bitcoin cryptocurrency',
                'ETH': 'Ethereum cryptocurrency',
                'BNB': 'Binance Coin'
            }
            
            search_term = search_terms.get(base_asset, f"{base_asset} cryptocurrency")
            news_articles = await self.news_service.get_news(search_term)
            
            # Extract relevant text from articles
            texts = []
            for article in news_articles:
                if (article.get('title')):
                    texts.append(article['title'])
                if (article.get('description')):
                    texts.append(article['description'])
            
            logger.info(f"Fetched {len(texts)} news items for {pair}")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {pair}: {e}")
            return []

    def generate_trades(self, signals: Dict) -> List[Dict]:
        """Generate trades based on signals and portfolio optimization"""
        trades = []
        portfolio_values = self._get_portfolio_values()
        
        for market_id, signal in signals.items():
            if signal['confidence'] > self.config['trading']['confidence_threshold']:
                # Calculate position size using Kelly criterion
                position_size = self._calculate_position_size(signal, portfolio_values)
                
                # Generate trade
                exchange, pair = market_id.split('_')
                trades.append({
                    'exchange': exchange,
                    'symbol': pair,
                    'size': position_size,
                    'direction': signal['direction'],
                    'type': 'market',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return trades

    async def execute_trades(self, trades: List[Dict]):
        """Execute trades across connected exchanges"""
        if not trades:
            return []

        results = []
        for trade in trades:
            try:
                client = self.exchange_clients[trade['exchange']]
                
                # Validate trade parameters
                if not all(k in trade for k in ['symbol', 'size', 'direction', 'type']):
                    logger.error(f"Invalid trade parameters: {trade}")
                    continue
                    
                if trade['direction'] > 0:
                    result = await client.create_buy_order(
                        symbol=trade['symbol'],
                        amount=trade['size'],
                        order_type=trade['type']
                    )
                else:
                    result = await client.create_sell_order(
                        symbol=trade['symbol'],
                        amount=trade['size'],
                        order_type=trade['type']
                    )
                
                if result:
                    results.append(result)
                    logger.info(f"Trade executed: {trade['symbol']} {'BUY' if trade['direction'] > 0 else 'SELL'}")
                    
            except Exception as e:
                logger.error(f"Trade execution error: {str(e)}")
                continue
        
        return results

    def _calculate_position_size(self, signal: Dict, portfolio_values: Dict) -> float:
        """Calculate position size with minimum quantity enforcement"""
        try:
            # Get minimum position value with safety margin
            min_position_value = self.config.get('trading', {}).get('min_trade_amount', 15.0)
            
            # Calculate Kelly fraction based on signal
            kelly_fraction = signal['confidence'] * signal['magnitude']
            
            # Get portfolio limits
            max_position = min(
                self.config.get('trading', {}).get('max_position_size', 0.2),
                portfolio_values['total'] * self.config['risk']['position_limit']
            )
            
            # Calculate base position size
            position_value = min(kelly_fraction * portfolio_values['total'], max_position)
            position_value = max(position_value, min_position_value)  # Enforce minimum
            
            # Additional safety check
            if position_value > portfolio_values['total']:
                position_value = 0.0  # Don't trade if insufficient funds
                
            logger.info(f"Calculated position value: ${position_value:.2f}")
            return position_value
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _get_portfolio_values(self) -> Dict:
        """Get current portfolio values with type validation"""
        if not self.portfolio:
            initial_balance = float(self.config.get('trading', {}).get('initial_balance', 1000.0))
            self.portfolio = {
                'total': initial_balance,
                'positions': {},
                'values': {'CASH': initial_balance}
            }
            
        # Ensure all values are numeric
        try:
            values = {}
            for k, v in self.portfolio.get('values', {}).items():
                try:
                    values[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid value for {k}: {v}")
                    continue
            
            self.portfolio['values'] = values
            
            return {
                'total': sum(values.values()),
                'positions': {k: float(v) for k, v in self.portfolio.get('positions', {}).items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio values: {e}")
            return {'total': 0.0, 'positions': {}}

    async def set_portfolio(self, portfolio: Dict):
        """Set current portfolio state"""
        self.portfolio = portfolio
        await self._update_portfolio_state()

    async def _update_portfolio_state(self):
        """Update portfolio state with proper type handling"""
        if not self.portfolio:
            return
            
        try:
            updated_values = {}
            for symbol, quantity in self.portfolio['positions'].items():
                try:
                    quantity = float(quantity)
                    for client in self.exchange_clients.values():
                        try:
                            price = await client.get_price(symbol)
                            if price > 0:
                                updated_values[symbol] = quantity * price
                                break
                        except:
                            continue
                except (TypeError, ValueError):
                    logger.warning(f"Invalid quantity for {symbol}: {quantity}")
                    continue
            
            # Add cash position if exists
            if 'CASH' in self.portfolio.get('values', {}):
                try:
                    updated_values['CASH'] = float(self.portfolio['values']['CASH'])
                except (TypeError, ValueError):
                    updated_values['CASH'] = 0.0
            
            self.portfolio['values'] = updated_values
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    def update_risk_metrics(self) -> Dict:
        """Calculate current risk metrics"""
        if not self.portfolio:
            return {}
        
        risk_metrics = self.risk_manager.calculate_risk_metrics(self.portfolio)
        
        # Add additional metrics
        risk_metrics.update({
            'exposure': self._calculate_exposure(),
            'concentration': self._calculate_concentration()
        })
        
        return risk_metrics

    def _calculate_exposure(self) -> float:
        """Calculate current market exposure with zero handling"""
        if not self.portfolio or not self.portfolio.get('values'):
            return 0.0
            
        total_value = sum(self.portfolio['values'].values())
        if total_value == 0:
            return 0.0
            
        abs_positions = sum(abs(v) for v in self.portfolio['values'].values())
        return abs_positions / total_value

    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration with proper numeric handling"""
        if not self.portfolio or 'values' not in self.portfolio:
            return 0.0
            
        # Convert all values to float and filter out non-numeric
        try:
            values = {}
            for k, v in self.portfolio['values'].items():
                try:
                    values[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid portfolio value for {k}: {v}")
                    continue
            
            if not values:
                return 0.0
                
            total = sum(values.values())
            if total == 0:
                return 0.0
                
            weights = np.array([v/total for v in values.values()])
            return float(np.sum(weights * weights))
            
        except Exception as e:
            logger.error(f"Error calculating concentration: {e}")
            return 0.0

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate trading symbol across exchanges"""
        for exchange in self.exchange_clients.values():
            if await exchange.has_symbol(symbol):
                return True
        return False

    # Add the missing risk reduction method
    async def _reduce_exposure(self):
        """Reduce portfolio exposure when risk limits are exceeded"""
        try:
            if not self.portfolio:
                return
                
            # Sort positions by size (largest first)
            positions = sorted(
                self.portfolio['positions'].items(),
                key=lambda x: self.portfolio['values'].get(x[0], 0),
                reverse=True
            )
            
            # Calculate target reduction (reduce by 25%)
            current_exposure = self._calculate_exposure()
            target_exposure = self.config['risk']['leverage_limit'] * 0.75
            
            for symbol, amount in positions:
                if current_exposure <= target_exposure:
                    break
                    
                # Calculate reduction amount
                current_value = self.portfolio['values'].get(symbol, 0)
                reduction_pct = min(0.5, (current_exposure - target_exposure) / current_exposure)
                reduction_amount = amount * reduction_pct
                
                # Create reduction trade
                if reduction_amount > 0:
                    trade = {
                        'exchange': next(iter(self.exchange_clients.keys())),  # Use first exchange
                        'symbol': symbol,
                        'size': reduction_amount,
                        'direction': -1,  # Sell/reduce
                        'type': 'market',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Execute reduction trade
                    logger.warning(f"Reducing exposure: Selling {reduction_amount:.4f} {symbol}")
                    await self.execute_trades([trade])
                    
                    # Update exposure
                    current_exposure = self._calculate_exposure()
            
            logger.info(f"Exposure reduction complete. New exposure: {current_exposure:.2%}")
            
        except Exception as e:
            logger.error(f"Error reducing exposure: {e}")

    # Add robo advisor methods
    async def register_client(self, client_id: str, profile: Dict) -> None:
        """Register a new client for robo advisory"""
        await self.robo_service.register_client(client_id, profile)
        self.client_profiles[client_id] = profile

    async def get_portfolio_recommendation(self, client_id: str) -> Dict:
        """Get personalized portfolio recommendation"""
        return await self.robo_service.get_portfolio_recommendation(client_id)

    async def analyze_client_portfolio(self, client_id: str) -> Dict:
        """Analyze client portfolio including ESG and tax considerations"""
        return await self.robo_service.analyze_portfolio(client_id)

    async def generate_client_trades(self, client_id: str) -> List[Dict]:
        """Generate trades for client portfolio rebalancing"""
        robo_trades = await self.robo_service.generate_trades(client_id)
        
        # Combine with market inefficiency trades if needed
        market_data = await self.get_market_data()
        signals = await self.detect_inefficiencies(market_data)
        system_trades = self.generate_trades(signals)
        
        # Merge and validate trades
        return self._merge_trade_recommendations(robo_trades, system_trades)

    def _merge_trade_recommendations(self, robo_trades: List[Dict], 
                                   system_trades: List[Dict]) -> List[Dict]:
        """Merge and prioritize trade recommendations"""
        merged = []
        seen_symbols = set()
        
        # Prioritize robo advisor trades (rebalancing, tax-loss harvesting)
        for trade in robo_trades:
            merged.append(trade)
            seen_symbols.add(trade['symbol'])
        
        # Add system trades for symbols not covered by robo trades
        for trade in system_trades:
            if trade['symbol'] not in seen_symbols:
                merged.append(trade)
        
        return merged

    async def run(self):
        """Main trading loop"""
        try:
            logger.info("Starting trading system initialization...")
            await self.initialize()
            
            # Add heartbeat logging
            last_heartbeat = 0
            HEARTBEAT_INTERVAL = 10  # seconds
            
            iteration = 0
            while self.is_running:
                try:
                    current_time = datetime.now().timestamp()
                    
                    # Log heartbeat
                    if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
                        logger.info("System heartbeat - Trading system active")
                        last_heartbeat = current_time
                    
                    iteration += 1
                    logger.info(f"\n{'='*50}\nTrading Iteration {iteration}\n{'='*50}")
                    
                    # 1. Get market data
                    logger.info("Fetching market data...")
                    market_data = await self.get_market_data()
                    logger.info(f"Received data for {len(market_data)} exchanges")
                    
                    #  2. Detect trading opportunities
                    logger.info("Analyzing market inefficiencies...")
                    signals = await self.detect_inefficiencies(market_data)
                    if signals:
                        logger.info(f"Detected {len(signals)} trading signals")
                        for market_id, signal in signals.items():
                            logger.info(f"Signal for {market_id}: "
                                      f"confidence={signal['confidence']:.2f}, "
                                      f"direction={'LONG' if signal['direction'] > 0 else 'SHORT'}")
                    
                    # 3. Generate system trades
                    logger.info("Generating system trades...")
                    system_trades = self.generate_trades(signals)
                    if system_trades:
                        logger.info(f"Generated {len(system_trades)} system trades")
                    
                    # 4. Handle robo-advisory tasks
                    logger.info("Processing robo-advisory tasks...")
                    for client_id in self.client_profiles:
                        logger.info(f"Generating trades for client {client_id}")
                        robo_trades = await self.generate_client_trades(client_id)
                        if robo_trades:
                            logger.info(f"Executing {len(robo_trades)} robo-advisory trades")
                            results = await self.execute_trades(robo_trades)
                            logger.info(f"Robo trades execution complete: {len(results)} orders filled")
                    
                    # 5. Execute system trades
                    if system_trades:
                        logger.info(f"Executing {len(system_trades)} system trades...")
                        results = await self.execute_trades(system_trades)
                        logger.info(f"System trades execution complete: {len(results)} orders filled")
                    
                    # 6. Update portfolio state
                    logger.info("Updating portfolio state...")
                    await self._update_portfolio_state()
                    
                    # 7. Check risk metrics
                    logger.info("Calculating risk metrics...")
                    risk_metrics = self.update_risk_metrics()
                    risk_config = self.config.get('risk', {})
                    
                    # Log risk metrics
                    logger.info("\nRisk Metrics:")
                    logger.info(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0.0):.2%}")
                    logger.info(f"VaR: {risk_metrics.get('var', 0.0):.2%}")
                    logger.info(f"Exposure: {float(risk_metrics.get('exposure', 0.0)):.2%}")
                    logger.info(f"Concentration: {risk_metrics.get('concentration', 0.0):.2%}")
                    
                    # Check risk limits with defaults
                    if (risk_metrics.get('max_drawdown', 0) > risk_config.get('max_drawdown', 0.10) or
                        risk_metrics.get('var', 0) > risk_config.get('var_limit', 0.02) or
                        risk_metrics.get('exposure', 0) > risk_config.get('leverage_limit', 1.0)):
                        logger.warning("⚠️ Risk limit exceeded, reducing exposure")
                        await self._reduce_exposure()
                    
                    # 8. Wait for next iteration
                    interval = self.config.get('trading', {}).get('loop_interval', 60)
                    logger.info(f"\nWaiting {interval} seconds until next iteration...\n")
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(5)  # Brief pause before retrying
                    
        except KeyboardInterrupt:
            logger.info("\nReceived shutdown signal. Cleaning up...")
        finally:
            logger.info("Initiating shutdown sequence...")
            await self.shutdown()
            logger.info("Trading system shutdown complete")

    async def _setup_exchange(self, exchange_config: Dict) -> None:
        """Setup single exchange connection"""
        try:
            exchange_type = exchange_config.get('name', '').lower()
            
            if exchange_type == 'binance':
                from services.exchanges.binance import BinanceClient
                
                # Get validated credentials
                creds = self.config_manager.get_exchange_credentials('binance')
                
                # Use singleton instance
                client = await BinanceClient.get_instance({
                    'api_key': creds['api_key'],
                    'api_secret': creds['api_secret'],
                    'test_mode': exchange_config.get('test_mode', True)
                })
                
                self.exchange_clients[exchange_type] = client
                logger.info(f"Using shared Binance client instance")
                
            else:
                logger.warning(f"Unsupported exchange type: {exchange_type}")
                
        except Exception as e:
            logger.error(f"Failed to setup {exchange_type}: {str(e)}")
            raise

if __name__ == "__main__":
    # Windows-specific event loop policy
    if sys.platform.startswith('win'):
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Load config 
    config_path = "config/trading.yaml"
    system = None
    loop = None
    
    try:
        # Set up clean shutdown handler
        def handle_shutdown(signum, frame):
            logger.info("\nShutdown signal received...")
            if loop and system:
                loop.run_until_complete(system.shutdown())
            else:
                sys.exit(0)
                
        # Register signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Create and run system
        system = TradingSystem(config_path)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(system.run())
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if system and loop:
            loop.run_until_complete(system.shutdown())
        sys.exit(1)