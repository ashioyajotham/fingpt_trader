"""Backtesting system that matches live trading behavior"""

import sys
import os
import logging
import yaml
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from services.trading.robo_service import RoboService 
from models.portfolio.rebalancing import Portfolio
from services.exchanges.binance import BinanceClient

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting engine that simulates live trading"""
    
    def __init__(self, config_path: str):
        """Initialize backtester with config"""
        self.config = self._load_config(config_path)
        
        # Initialize exchange client
        exchange_config = {
            'api_key': self.config['exchange']['api_key'],
            'api_secret': self.config['exchange']['api_secret'],
            'testnet': self.config['exchange'].get('test_mode', True),
            'options': self.config['exchange'].get('options', {})
        }
        self.exchange = BinanceClient(**exchange_config)
        self.portfolio = Portfolio(10000.0)
        self.service = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.pair_format_map = {}  # Maps config pairs to exchange format
        
    def _load_config(self, path: str) -> dict:
        """Load config from yaml file"""
        with open(path) as f:
            return yaml.safe_load(f)
            
    def _convert_pair_format(self, pair: str, reverse: bool = False) -> str:
        """Convert between config pair format and exchange format"""
        if reverse:
            return pair.replace('/', '')  # "BTC/USDT" -> "BTCUSDT"
        return '/'.join([pair[:3], pair[3:]])  # "BTCUSDT" -> "BTC/USDT"
            
    async def _detect_available_date_range(self, pair: str) -> Tuple[datetime, datetime]:
        """Detect the most recent available data range from exchange"""
        try:
            # Try to get recent data (last 5 days)
            now = datetime.now()
            recent_start = now - timedelta(days=5)
            
            recent_data = await self.exchange.get_historical_klines(
                symbol=pair,
                interval='1d',
                start_str=recent_start.strftime('%Y-%m-%d'),
                end_str=now.strftime('%Y-%m-%d')
            )
            
            if not recent_data:
                raise ValueError(f"No data available for {pair}")
                
            # Get the most recent timestamp
            latest = pd.to_datetime(recent_data[-1][0], unit='ms')
            
            # Try to get data from 60 days before latest
            historical_start = latest - timedelta(days=60)
            historical = await self.exchange.get_historical_klines(
                symbol=pair,
                interval='1d',
                start_str=historical_start.strftime('%Y-%m-%d'),
                end_str=latest.strftime('%Y-%m-%d')
            )
            
            if historical:
                earliest = pd.to_datetime(historical[0][0], unit='ms')
                return earliest, latest
            
            # If no historical data, return recent range
            return latest - timedelta(days=30), latest
            
        except Exception as e:
            logger.error(f"Error detecting date range: {e}")
            # Return a safe default range
            now = datetime.now()
            return now - timedelta(days=30), now

    async def load_data(self, start_date: datetime, end_date: datetime):
        """Load historical market data with automatic date adjustment"""
        logger.info("Loading historical market data...")
        
        # Detect available date range from first pair
        first_pair = self._convert_pair_format(self.config['trading']['pairs'][0], reverse=True)
        available_start, available_end = await self._detect_available_date_range(first_pair)
        
        # Adjust requested date range to available data
        if start_date > available_end or end_date < available_start:
            logger.warning(f"Requested date range ({start_date.date()} to {end_date.date()}) "
                         f"outside available data range ({available_start.date()} to {available_end.date()})")
            start_date = available_end - timedelta(days=30)
            end_date = available_end
            logger.info(f"Adjusted to most recent available range: {start_date.date()} to {end_date.date()}")

        # Load price data for each pair
        for pair in self.config['trading']['pairs']:
            # Convert pair format for exchange API
            exchange_pair = self._convert_pair_format(pair, reverse=True)
            self.pair_format_map[pair] = exchange_pair
            
            try:
                self.data[pair] = await self._load_historical_data(
                    pair=exchange_pair,
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
                logger.info(f"Loaded data for {pair}")
            except Exception as e:
                logger.error(f"Failed to load data for {pair}: {str(e)}")
                continue
        
    async def _load_historical_data(self, pair: str, start: datetime, end: datetime, 
                                  interval: str) -> pd.DataFrame:
        """Load historical OHLCV data from exchange"""
        try:
            # Ensure we're not requesting future data
            end = min(end, datetime.now())
            if start > end:
                raise ValueError("Start date must be before end date")

            data = await self.exchange.get_historical_klines(
                symbol=pair,
                interval=interval,
                start_str=start.strftime('%Y-%m-%d'),
                end_str=end.strftime('%Y-%m-%d')
            )
            
            if not data:
                logger.warning(f"No data returned for {pair}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {str(e)}")
            return pd.DataFrame()

    async def simulate_trading(self, current_time: datetime) -> None:
        """Simulate one trading interval"""
        try:
            # Get current market data
            market_data = {}
            signals = {}
            
            for pair in self.config['trading']['pairs']:
                exchange_pair = self.pair_format_map.get(pair)
                if not exchange_pair:
                    logger.warning(f"No exchange pair mapping for {pair}")
                    continue
                    
                if pair not in self.data:
                    logger.warning(f"No historical data for {pair}")
                    continue
                    
                try:
                    current_price = float(self.data[pair].loc[current_time, 'close'])
                    market_data[pair] = {
                        'price': current_price,
                        'volume': float(self.data[pair].loc[current_time, 'volume']),
                        'high': float(self.data[pair].loc[current_time, 'high']),
                        'low': float(self.data[pair].loc[current_time, 'low'])
                    }
                except KeyError:
                    logger.debug(f"No data for {pair} at {current_time}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {pair} data: {str(e)}")
                    continue

            # Skip if no market data available
            if not market_data:
                logger.warning(f"No market data available for {current_time}")
                return
            
            # Update portfolio prices first
            self.portfolio.update_prices(market_data)
            
            # Generate signals and execute trades
            for pair, data in market_data.items():
                # Get current position
                position = self.portfolio.get_position(pair)
                
                # Generate trading signal
                signal = await self.service.analyze_position(
                    pair=pair,
                    price=data['price'],
                    position=position
                )
                signals[pair] = signal
                
                # Execute trades based on signals
                if signal:
                    await self._execute_trade(pair, signal, data['price'])
                    
            # Log portfolio state
            await self._log_state(current_time, market_data, signals)
            
        except Exception as e:
            logger.error(f"Error in simulation at {current_time}: {str(e)}")
            
    async def _execute_trade(self, pair: str, signal: str, price: float):
        """Execute a simulated trade"""
        if signal == 'BUY':
            # Calculate position size based on available cash
            size = self.portfolio.get_position_size(
                cash=self.portfolio.cash,
                price=price
            )
            if size > 0:
                logger.info(f"TEST TRADE: Bought {size:.6f} {pair} at ${price:.2f}")
                await self.portfolio.buy(pair, size, price)
                
        elif signal == 'SELL':
            size = self.portfolio.get_position(pair)
            if size > 0:
                logger.info(f"TEST TRADE: Sold {size:.6f} {pair} at ${price:.2f}")
                await self.portfolio.sell(pair, size, price)

    async def run(self, start_date: datetime, end_date: datetime):
        """Run backtest simulation"""
        # Validate date range
        now = datetime.now()
        if start_date > now or end_date > now:
            logger.warning("Adjusting date range to use historical data only")
            if start_date > now:
                start_date = now - timedelta(days=365)  # Default to last year
            if end_date > now:
                end_date = now
                
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        # Initialize exchange client first
        await self.exchange.initialize()
        
        # Initialize services
        self.service = RoboService(self.config)
        await self.service.start()
        
        # Load data
        await self.load_data(start_date, end_date)
        
        # Verify data availability
        if not any(len(df) > 0 for df in self.data.values()):
            logger.error("No historical data available for the specified date range")
            return
            
        # Simulate trading
        logger.info("Simulating trades...")
        current_time = start_date
        while current_time <= end_date:
            if current_time.weekday() < 5:  # Only trade on weekdays
                await self.simulate_trading(current_time)
            current_time += timedelta(days=1)
            
        # Generate final results
        await self._generate_results()

    async def cleanup(self):
        """Cleanup resources"""
        if self.exchange:
            await self.exchange.cleanup()
        if self.service:
            await self.service.cleanup()

    async def _log_state(self, timestamp: datetime, market_data: dict, signals: dict):
        """Log current simulation state"""
        total_value = self.portfolio.total_value()
        
        logger.info("\n=== Status Update ===")
        logger.info(f"Time: {timestamp}")
        logger.info("")
        
        for pair in self.config['trading']['pairs']:
            logger.info(f"{pair}:")
            logger.info(f"  Price: ${market_data[pair]['price']:.2f}")
            logger.info(f"  Volume: {market_data[pair]['volume']:.2f}")
            logger.info(f"  Holdings: {self.portfolio.get_position(pair):.6f}")
            logger.info(f"  Value: ${self.portfolio.get_position_value(pair):.2f}")
            logger.info(f"  Signal: {signals[pair]}")
            logger.info("")
            
        logger.info("Portfolio Status:")
        logger.info(f"  USDT Balance: ${self.portfolio.cash:.2f}")
        logger.info(f"  Total Value: ${total_value:.2f}")
        logger.info("==================\n")

    async def _generate_results(self):
        """Generate backtest results summary"""
        results = {
            'total_trades': self.portfolio.total_trades,
            'total_return': (self.portfolio.total_value() / 10000.0) - 1,
            'sharpe_ratio': self.portfolio.calculate_sharpe_ratio(),
            'max_drawdown': self.portfolio.calculate_max_drawdown(),
            'win_rate': self.portfolio.calculate_win_rate(),
            'profit_factor': self.portfolio.calculate_profit_factor()
        }
        
        logger.info("\n=== Backtest Results ===")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {results['win_rate']:.1%}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FinGPT Backtesting System')
    
    # Calculate default date range based on today
    today = datetime.now()
    default_end = today - timedelta(days=1)  # Yesterday
    default_start = default_end - timedelta(days=30)  # 30 days before yesterday
    
    parser.add_argument(
        '--start-date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=default_start.strftime('%Y-%m-%d'),
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=default_end.strftime('%Y-%m-%d'),
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=None,
        help='Comma-separated list of trading pairs (e.g. BTC/USDT,ETH/USDT)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(project_root, "config", "trading.yaml"),
        help='Path to config file'
    )
    
    parser.add_argument(
        '--initial-balance',
        type=float,
        default=10000.0,
        help='Initial portfolio balance'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Only validate that start is before end
    if args.start_date > args.end_date:
        logger.warning("Start date must be before end date, adjusting range")
        args.start_date = args.end_date - timedelta(days=30)
    
    return args

async def main():
    """Run backtest"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    # Load config and override with command line args
    engine = BacktestEngine(args.config)
    if args.pairs:
        engine.config['trading']['pairs'] = args.pairs.split(',')
    engine.portfolio = Portfolio(args.initial_balance)
    
    logger.info(f"""Backtest Configuration:
    Date Range: {args.start_date.date()} to {args.end_date.date()}
    Trading Pairs: {engine.config['trading']['pairs']}
    Initial Balance: ${args.initial_balance:,.2f}
    Using Testnet: {engine.config['exchange'].get('test_mode', True)}
    """)
    
    try:
        await engine.run(args.start_date, args.end_date)
    finally:
        await engine.cleanup()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Fix for Windows event loop
    if sys.platform == 'win32':
        from asyncio import WindowsSelectorEventLoopPolicy, set_event_loop_policy
        set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
