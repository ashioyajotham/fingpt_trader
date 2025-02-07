"""Backtesting system that matches live trading behavior"""

import sys
import os
import logging
import yaml
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

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
            
    async def load_data(self, start_date: datetime, end_date: datetime):
        """Load historical market data"""
        logger.info("Loading historical market data...")
        
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
        data = await self.exchange.get_historical_klines(
            symbol=pair,
            interval=interval,
            start_str=start.isoformat(),
            end_str=end.isoformat()
        )
        
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
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        # Initialize exchange client first
        await self.exchange.initialize()
        
        # Initialize services
        self.service = RoboService(self.config)
        await self.service.start()
        
        # Load data
        await self.load_data(start_date, end_date)
        
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
        
async def main():
    """Run backtest"""
    config_path = os.path.join(project_root, "config", "trading.yaml")
    engine = BacktestEngine(config_path)
    
    start = datetime(2025, 1, 1)
    end = datetime.now()
    
    try:
        await engine.run(start, end)
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
