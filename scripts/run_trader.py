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
from typing import Dict, Optional, List
import yaml
import platform

import os
from dotenv import load_dotenv  # Add this import
import time
import datetime
import pandas as pd  # Add missing import

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
        # Add test account settings
        self.test_balance = {
            'USDT': 10000.0,  # Initial test funding
            'BTC': 0.0,
            'ETH': 0.0
        }
        self.trade_history = []
        self.analysis_data = {}
        self.price_history = {pair: pd.DataFrame() for pair in self.monitored_pairs}
        self.technical = self  # Use self as technical analyzer
        self.position_tracking = {
            'BTCUSDT': {'entry_price': 0, 'entry_time': None},
            'ETHUSDT': {'entry_price': 0, 'entry_time': None}
        }

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
                try:
                    # Get current ticker and candles
                    ticker = await self.exchange.get_ticker(pair)
                    candles = await self.exchange.get_candles(pair, interval='5m', limit=100)
                    
                    if ticker and ticker.get('price'):
                        # Update current price data
                        self.price_data[pair] = {
                            'price': float(ticker['price']),
                            'timestamp': ticker['timestamp']
                        }
                        
                        # Update price history with proper column handling
                        if candles:
                            df = pd.DataFrame(candles)
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                        'taker_buy_quote', 'ignore']
                            # Convert numeric columns
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = df[col].astype(float)
                            
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            self.price_history[pair] = df

                except Exception as e:
                    logger.error(f"Failed to update {pair} market data: {e}")
                    continue

        except Exception as e:
            logger.error(f"Market data update failed: {e}")

    def calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        return prices.rolling(period).mean().iloc[-1]
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
        
    def detect_trend(self, prices: pd.Series, sma_short: int = 20, sma_long: int = 50) -> str:
        """Detect market trend using moving averages"""
        short_ma = prices.rolling(sma_short).mean().iloc[-1]
        long_ma = prices.rolling(sma_long).mean().iloc[-1]
        
        if short_ma > long_ma:
            return "BULLISH"
        elif short_ma < long_ma:
            return "BEARISH"
        return "NEUTRAL"

    async def analyze_market(self):
        """Market analysis using technical indicators and strategies"""
        try:
            for pair in self.monitored_pairs:
                if pair not in self.price_data or pair not in self.price_history:
                    continue
                    
                df = self.price_history[pair]
                if len(df) < 50:  # Need enough data for analysis
                    continue
                    
                closes = df['close']
                current_price = self.price_data[pair]['price']
                
                # Technical Analysis
                rsi = self.calculate_rsi(closes)
                trend = self.detect_trend(closes)
                sma_20 = self.calculate_sma(closes, 20)
                sma_50 = self.calculate_sma(closes, 50)
                
                # Calculate holding period
                holding_period = 0
                if self.test_balance[pair.replace('USDT', '')] > 0:
                    if self.position_tracking[pair]['entry_time']:
                        holding_period = (datetime.datetime.now() - 
                                       self.position_tracking[pair]['entry_time']).days

                # Get RoboService strategy signal with proper position info
                robo_signal = await self.robo_service.analyze_position(
                    pair=pair,
                    price=current_price,
                    position={
                        'size': self.test_balance[pair.replace('USDT', '')],
                        'entry_price': self.position_tracking[pair]['entry_price'],
                        'holding_period': holding_period
                    }
                )
                
                # Combine technical and strategy signals
                signal = None
                tech_signal = None
                
                # Technical signal generation
                if trend == "BULLISH" and (
                    (rsi < 30) or  # Oversold condition
                    (current_price > sma_20 and rsi < 60)  # Uptrend confirmation
                ):
                    tech_signal = "BUY"
                elif trend == "BEARISH" and (
                    (rsi > 70) or  # Overbought condition
                    (current_price < sma_20 and rsi > 40)  # Downtrend confirmation
                ):
                    tech_signal = "SELL"
                
                # Combine signals - need both technical and strategy agreement
                if tech_signal and robo_signal and tech_signal == robo_signal:
                    signal = tech_signal
                
                self.analysis_data[pair] = {
                    'price': current_price,
                    'trend': trend,
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'tech_signal': tech_signal,
                    'robo_signal': robo_signal,
                    'signal': signal
                }
                
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")

    async def analyze_market(self):
        """Market analysis using technical indicators and strategies"""
        try:
            for pair in self.monitored_pairs:
                if pair not in self.price_data or pair not in self.price_history:
                    continue
                    
                df = self.price_history[pair]
                if len(df) < 50:  # Need enough data for analysis
                    continue
                    
                closes = df['close']
                current_price = self.price_data[pair]['price']
                
                # Technical Analysis
                rsi = self.calculate_rsi(closes)
                trend = self.detect_trend(closes)
                sma_20 = self.calculate_sma(closes, 20)
                sma_50 = self.calculate_sma(closes, 50)
                
                # Get position info for strategy
                base_asset = pair.replace('USDT', '')
                position_size = self.test_balance[base_asset]
                entry_price = self.position_tracking[pair]['entry_price']
                holding_period = 0
                
                if position_size > 0 and self.position_tracking[pair]['entry_time']:
                    holding_period = (datetime.datetime.now() - 
                                   self.position_tracking[pair]['entry_time']).days
                    
                # Calculate unrealized PnL    
                unrealized_pnl = 0
                if position_size > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price

                # Get strategy signal with complete position info
                robo_signal = await self.robo_service.analyze_position(
                    pair=pair,
                    price=current_price,
                    position={
                        'size': position_size,
                        'entry_price': entry_price,
                        'holding_period': holding_period,
                        'unrealized_pnl': unrealized_pnl
                    }
                )

                # Generate technical signal
                tech_signal = None
                if trend == "BULLISH" and rsi < 30:  # Simplified conditions
                    tech_signal = "BUY"
                elif trend == "BEARISH" and rsi > 70:
                    tech_signal = "SELL"

                # Final signal if either condition is met
                signal = robo_signal if robo_signal else tech_signal

                self.analysis_data[pair] = {
                    'price': current_price,
                    'trend': trend,
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'tech_signal': tech_signal,
                    'robo_signal': robo_signal,
                    'signal': signal
                }
                
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")

    def _generate_signal(self, price: float, sma_5: float, sma_20: float) -> Optional[str]:
        """Generate trading signal based on SMAs"""
        if sma_5 > sma_20 and price > sma_5:
            return 'BUY'
        elif sma_5 < sma_20 and price < sma_5:
            return 'SELL'
        return None

    def _combine_signals(self, signals: List[str]) -> Optional[str]:
        """Combine signals from different strategies"""
        if not signals:
            return None
            
        # If any strategy says SELL, we sell
        if 'SELL' in signals:
            return 'SELL'
            
        # Need unanimous BUY signals to buy
        if all(s == 'BUY' for s in signals):
            return 'BUY'
            
        return None

    async def execute_test_trade(self, pair: str, side: str, signal_type: str):
        """Execute a simulated trade"""
        try:
            current_price = float(self.price_data[pair]['price'])
            base_asset = pair.replace('USDT', '')
            
            # Calculate trade size (1% of USDT balance)
            trade_size_usdt = self.test_balance['USDT'] * 0.01
            
            if side == 'BUY' and self.test_balance['USDT'] >= trade_size_usdt:
                quantity = trade_size_usdt / current_price
                self.test_balance['USDT'] -= trade_size_usdt
                self.test_balance[base_asset] += quantity
                
                trade = {
                    'timestamp': datetime.datetime.now(),
                    'pair': pair,
                    'side': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'value_usdt': trade_size_usdt,
                    'signal': signal_type
                }
                self.trade_history.append(trade)
                logger.info(f"TEST TRADE: Bought {quantity:.6f} {base_asset} at ${current_price:.2f}")
                
                self.position_tracking[pair] = {
                    'entry_price': current_price,
                    'entry_time': datetime.datetime.now()
                }
                
            elif side == 'SELL' and self.test_balance[base_asset] > 0:
                quantity = self.test_balance[base_asset] * 0.1  # Sell 10% of holdings
                value_usdt = quantity * current_price
                self.test_balance['USDT'] += value_usdt
                self.test_balance[base_asset] -= quantity
                
                trade = {
                    'timestamp': datetime.datetime.now(),
                    'pair': pair,
                    'side': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'value_usdt': value_usdt,
                    'signal': signal_type
                }
                self.trade_history.append(trade)
                logger.info(f"TEST TRADE: Sold {quantity:.6f} {base_asset} at ${current_price:.2f}")
                
                self.position_tracking[pair] = {
                    'entry_price': 0,
                    'entry_time': None
                }
                
        except Exception as e:
            logger.error(f"Test trade execution failed: {e}")

    async def check_trading_conditions(self):
        """Check and execute trades based on conditions"""
        for pair in self.monitored_pairs:
            if pair not in self.analysis_data:
                continue
                
            signal = self.analysis_data[pair]['signal']
            if signal:
                await self.execute_test_trade(pair, signal, 'SMA_CROSS')

    async def print_status_update(self):
        """Print periodic status update"""
        now = time.time()
        if now - self.last_status_update >= self.status_interval:
            self.last_status_update = now
            total_value_usdt = self.test_balance['USDT']
            
            logger.info("\n=== Status Update ===")
            logger.info(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Market Data and Analysis
            for pair in self.monitored_pairs:
                if pair not in self.price_data:
                    continue
                    
                base_asset = pair.replace('USDT', '')
                current_price = self.price_data[pair]['price']
                asset_value_usdt = current_price * self.test_balance[base_asset]
                total_value_usdt += asset_value_usdt
                
                logger.info(f"\n{pair}:")
                logger.info(f"  Price: ${current_price:.2f}")
                logger.info(f"  Holdings: {self.test_balance[base_asset]:.6f} {base_asset}")
                logger.info(f"  Value: ${asset_value_usdt:.2f}")
                
                if pair in self.analysis_data:
                    analysis = self.analysis_data[pair]
                    logger.info(f"  Trend: {analysis['trend']}")
                    logger.info(f"  RSI: {analysis['rsi']:.1f}")
                    logger.info(f"  Technical Signal: {analysis['tech_signal']}")
                    logger.info(f"  Strategy Signal: {analysis['robo_signal']}")
                    logger.info(f"  Final Signal: {analysis['signal']}")
            
            # Portfolio Status
            logger.info(f"\nPortfolio Status:")
            logger.info(f"  USDT Balance: ${self.test_balance['USDT']:.2f}")
            logger.info(f"  Total Value: ${total_value_usdt:.2f}")
            
            # System Status
            logger.info(f"\nSystem Status:")
            logger.info(f"  Running Time: {time.time() - self.start_time:.1f}s")
            logger.info(f"  Memory Usage: {self._get_memory_usage():.1f}MB")
            logger.info(f"  Trades Today: {len(self.trade_history)}")
            logger.info("==================\n")

    async def run(self):
        """Main trading loop"""
        try:
            self.start_time = time.time()
            await self.startup()
            
            while self.running:
                try:
                    # Update market data
                    await self.update_market_data()
                    
                    # Only proceed with analysis if we have market data
                    if self.price_data:
                        await self.analyze_market()
                        await self.check_trading_conditions()
                        await self.print_status_update()
                    else:
                        logger.warning("Skipping trading loop - no market data available")
                        
                    # Main loop interval
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(5)  # Back off on error
                    
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
