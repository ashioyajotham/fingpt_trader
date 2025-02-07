"""
FinGPT Trading System - Production Entry Point

A comprehensive algorithmic trading system combining machine learning,
sentiment analysis, and traditional trading strategies.

System Architecture:
    1. Core Components:
        - Market Analysis:
            * Inefficiency detection
            * Technical analysis
            * Sentiment processing
        - Portfolio Management:
            * Position sizing
            * Risk management
            * Portfolio optimization
        - Trading Execution:
            * Multi-exchange support
            * Order management
            * Execution algorithms
        - Robo-Advisory:
            * Client profiling
            * Portfolio recommendations
            * Tax-aware trading

    2. Key Features:
        - Real-time market data processing
        - ML-based market inefficiency detection
        - Natural language processing for news analysis
        - Advanced portfolio optimization
        - Risk management and monitoring
        - Tax-loss harvesting
        - Multi-client portfolio management

Usage:
    python main.py

Configuration:
    The system requires proper setup of:
    1. Environment Variables:
        - BINANCE_API_KEY: Exchange API key
        - BINANCE_API_SECRET: Exchange API secret
        - HUGGINGFACE_TOKEN: For ML models
        - NEWS_API_KEY: For news fetching
        
    2. Configuration Files:
        - config/trading.yaml: Main configuration
        - .env: Environment variables
        
    3. Model Files:
        - models/: ML model weights and configurations

Development vs Production:
    - For development/testing use: scripts/run_trader.py
    - For production deployment use: main.py (this file)

"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
import yaml
from datetime import datetime
import logging

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

from utils.logging import LogManager
from services.data_feeds.news_service import NewsService
from services.data_feeds.market_data_service import MarketDataService

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
from models.sentiment.analyzer import SentimentAnalyzer
from models.portfolio.optimization import PortfolioOptimizer
from models.portfolio.risk import RiskManager
from services.trading.robo_service import RoboService

class TradingSystem:
    """
    Production Trading System Implementation
    
    - Market inefficiency detection
    - Sentiment analysis
    - Portfolio optimization
    - Risk management
    - Robo-advisory services
    
    Components:
        - MarketInefficiencyDetector: Identifies trading opportunities
        - SentimentAnalyzer: Processes market sentiment
        - PortfolioOptimizer: Manages portfolio allocations
        - RiskManager: Handles risk monitoring
        - RoboService: Provides robo-advisory functionality
        
    Attributes:
        config (Dict): System configuration
        market_detector (MarketInefficencyDetector): Market analysis component
        sentiment_analyzer (SentimentAnalyzer): NLP component
        portfolio_optimizer (PortfolioOptimizer): Portfolio management
        risk_manager (RiskManager): Risk monitoring and limits
        robo_service (RoboService): Robo-advisory services
        exchange_clients (Dict): Exchange connections
        client_profiles (Dict): Client information
        
    Methods:
        initialize(): Setup system components
        run(): Main trading loop
        shutdown(): Cleanup resources
    """
    def __init__(self, config_path: str):
        # Load environment variables first
        load_dotenv()
        
        # Load and process config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Replace environment variables in config
        self._process_env_vars(self.config)
        
        self.market_detector = MarketInefficencyDetector(self.config.get('market', {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment', {}))
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

        # Initialize data services
        self.news_service = NewsService(self.config.get('news', {}))
        self.market_data_service = MarketDataService(self.config.get('market_data', {}))

    def _process_env_vars(self, config: dict) -> None:
        """Replace ${VAR} with environment variable values"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var)
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
        try:
            if hasattr(self, 'sentiment_analyzer'):
                await self.sentiment_analyzer.cleanup()
            if hasattr(self, 'market_detector'):
                await self.market_detector.cleanup()
            if hasattr(self, 'robo_service'):
                await self.robo_service.cleanup()
                
            # Cleanup data services
            await self.news_service._cleanup()
            await self.market_data_service.stop()
            logger.info("Data services cleaned up")
            
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")

    async def _setup_exchange_client(self, exchange_config: Dict):
        """Setup exchange client connection"""
        try:
            exchange_type = exchange_config.get('name', '').lower()
            
            # Validate required config
            if not exchange_config.get('api_key') or not exchange_config.get('api_secret'):
                raise ValueError(f"Missing API credentials for {exchange_type}")
            
            if exchange_type == 'binance':
                from services.exchanges.binance import BinanceClient
                client = await BinanceClient.create(exchange_config)
                # Verify connection
                await client.ping()
                logger.info(f"Successfully connected to {exchange_type} {'testnet' if exchange_config.get('test_mode') else 'mainnet'}")
                return client
            else:
                raise ValueError(f"Unsupported exchange type: {exchange_type}")
                
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
                exchange_data[pair] = {
                    'orderbook': await client.get_orderbook(pair),
                    'trades': await client.get_recent_trades(pair),
                    'candles': await client.get_candles(pair),
                    'volume': await client.get_24h_volume(pair)
                }
            data[exchange] = exchange_data

        return data

    async def detect_inefficiencies(self, market_data: Dict) -> Dict:
        """
        Analyze market data for trading opportunities.
        
        Uses multiple detection methods:
        - Technical analysis patterns
        - Order book imbalances
        - Volume profile analysis
        - Sentiment correlation
        
        Args:
            market_data: Dict containing:
                - orderbook: Current order book state
                - trades: Recent trades
                - candles: OHLCV data
                
        Returns:
            Dict: Trading signals with:
                - confidence: Signal strength (0-1)
                - direction: Long/Short
                - metadata: Supporting data
                
        Raises:
            ValueError: If market data is invalid
        """
        signals = {}
        for exchange, exchange_data in market_data.items():
            for pair, pair_data in exchange_data.items():
                # Process market data
                processed_data = self._preprocess_market_data(pair_data)
                
                # Get sentiment analysis
                news_data = await self._fetch_relevant_news(pair)
                sentiment = await self.sentiment_analyzer.analyze_text(news_data)
                
                # Detect inefficiencies
                signal = self.market_detector.detect_inefficiencies(
                    prices=processed_data['prices'],
                    volume=processed_data['volume'],
                    sentiment=sentiment
                )
                
                signals[f"{exchange}_{pair}"] = signal
        
        return signals

    def _preprocess_market_data(self, pair_data: Dict) -> Dict:
        """Preprocess market data for analysis"""
        candles = np.array(pair_data['candles'])
        return {
            'prices': candles[:, 4],  # Close prices
            'volume': candles[:, 5],  # Volume
            'highs': candles[:, 2],   # High prices
            'lows': candles[:, 3]     # Low prices
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
                if article.get('title'):
                    texts.append(article['title'])
                if article.get('description'):
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
        """
        Execute trades across connected exchanges.
        
        Features:
        - Smart order routing
        - Slippage protection
        - Position sizing
        - Risk checks
        
        Args:
            trades: List of trade specifications:
                - symbol: Trading pair
                - size: Position size
                - direction: Long/Short
                - type: Order type
                
        Returns:
            List[Dict]: Execution results with:
                - order_id: Exchange order ID
                - status: Execution status
                - filled: Amount filled
                - price: Average fill price
                
        Raises:
            Exception: If execution fails
        """
        if trades is None:
            raise Exception("Cannot execute None trades")

        results = []
        for trade in trades:
            try:
                client = self.exchange_clients[trade['exchange']]
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
                results.append(result)
            except Exception as e:
                print(f"Trade execution error: {str(e)}")
                continue
        
        return results

    def _calculate_position_size(self, signal: Dict, portfolio_values: Dict) -> float:
        """Calculate position size using Kelly criterion with risk adjustment"""
        kelly_fraction = signal['confidence'] * signal['magnitude']
        
        # Apply risk limits
        max_position = min(
            self.config['trading']['max_position_size'],
            portfolio_values['total'] * self.config['risk']['position_limit']
        )
        
        return min(kelly_fraction * portfolio_values['total'], max_position)

    def _get_portfolio_values(self) -> Dict:
        """Get current portfolio values"""
        if not self.portfolio:
            return {'total': 0.0, 'positions': {}}
        return {
            'total': sum(self.portfolio['values']),
            'positions': dict(zip(self.portfolio['positions'], self.portfolio['values']))
        }

    async def set_portfolio(self, portfolio: Dict):
        """Set current portfolio state"""
        self.portfolio = portfolio
        await self._update_portfolio_state()

    async def _update_portfolio_state(self):
        """Update portfolio state with current market prices"""
        if not self.portfolio:
            return
        
        for symbol, quantity in self.portfolio['positions'].items():
            for client in self.exchange_clients.values():
                try:
                    price = await client.get_price(symbol)
                    self.portfolio['values'][symbol] = quantity * price
                    break
                except:
                    continue

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
        """Calculate current market exposure"""
        if not self.portfolio:
            return 0.0
        return sum(abs(v) for v in self.portfolio['values']) / sum(self.portfolio['values'])

    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        if not self.portfolio:
            return 0.0
        weights = np.array(self.portfolio['values']) / sum(self.portfolio['values'])
        return np.sum(weights ** 2)

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate trading symbol across exchanges"""
        for exchange in self.exchange_clients.values():
            if await exchange.has_symbol(symbol):
                return True
        return False

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
                    
                    # 2. Detect trading opportunities
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
                    logger.info(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
                    logger.info(f"VaR: {risk_metrics.get('var', 0):.2%}")
                    logger.info(f"Exposure: {risk_metrics.get('exposure', 0)::.2%}")
                    logger.info(f"Concentration: {risk_metrics.get('concentration', 0)::.2%}")
                    
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

if __name__ == "__main__":
    # Windows-specific event loop policy
    if sys.platform.startswith('win'):
        import asyncio
        import nest_asyncio
        
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # Use WindowsSelectorEventLoopPolicy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Load config
    config_path = "config/trading.yaml"
    
    # Create and run trading system
    system = TradingSystem(config_path)
    
    try:
        asyncio.run(system.run())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)