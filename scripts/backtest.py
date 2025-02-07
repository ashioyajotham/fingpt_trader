import sys
from pathlib import Path
import asyncio
import argparse
import yaml
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from strategies.base_strategy import BaseStrategy
from strategies.tax_aware import TaxAwareStrategy
from models.portfolio.rebalancing import Portfolio
from models.client.profile import MockClientProfile

logger = logging.getLogger(__name__)

class BacktestPortfolio:
    """Portfolio simulator for backtesting"""
    def __init__(self, initial_capital: float = 10000.0):
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.nav_history = []
        
    def execute_trade(self, symbol: str, side: str, price: float, 
                     amount: float, timestamp: datetime) -> bool:
        """Execute trade with slippage simulation"""
        slippage = self._calculate_slippage(price, amount)
        executed_price = price * (1 + slippage if side == 'BUY' else -slippage)
        
        cost = amount * executed_price
        if side == 'BUY':
            if cost > self.cash:
                return False
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + amount
        else:
            if amount > self.positions.get(symbol, 0):
                return False
            self.cash += cost
            self.positions[symbol] = self.positions.get(symbol, 0) - amount
            
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'price': executed_price,
            'amount': amount,
            'cost': cost,
            'slippage': slippage
        })
        return True
        
    def _calculate_slippage(self, price: float, amount: float) -> float:
        """Simulate market impact and slippage"""
        base_slippage = 0.0001  # 1 bps
        impact = amount * price / 1_000_000  # Simple market impact model
        return base_slippage + impact

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0)
        
    def get_nav(self, prices: Dict[str, float]) -> float:
        """Calculate Net Asset Value"""
        positions_value = sum(
            self.positions.get(s, 0) * p 
            for s, p in prices.items()
        )
        return self.cash + positions_value

class Backtester:
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('trading', {}).get('initial_capital', 10000.0)
        self.portfolio = BacktestPortfolio(initial_capital=self.initial_capital)
        self.data = {}
        self.results = []
        self.stats = {}
        
    async def run(self, strategy: BaseStrategy, start_date: str, end_date: str):
        """Run backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Load data
        await self._load_data(start_date, end_date)
        
        # Run simulation
        await self._simulate_trading(strategy)
        
        # Calculate statistics
        self._calculate_stats()
        self._print_results()
        return self.stats

    async def _load_data(self, start_date: str, end_date: str):
        """Load historical price and market data"""
        logger.info("Loading historical data...")
        
        # Load OHLCV data for configured pairs
        pairs = self.config['trading']['pairs']
        for pair in pairs:
            self.data[pair] = await self._load_market_data(
                pair, start_date, end_date
            )
            
        # Load additional market data
        await self._load_sentiment_data(start_date, end_date)
        await self._load_fundamentals_data(start_date, end_date)

    async def _load_market_data(self, pair: str, start: str, end: str) -> pd.DataFrame:
        """Generate realistic market data"""
        dates = pd.date_range(start=start, end=end, freq='1min')
        df = pd.DataFrame(index=dates)
        
        # Generate realistic price movements
        if pair == 'BTCUSDT':
            base_price = 97000.0
            volatility = 0.0002  # Annualized volatility
            drift = 0.00005     # Annualized drift
        elif pair == 'ETHUSDT':
            base_price = 2700.0
            volatility = 0.0003
            drift = 0.00004
        else:
            base_price = 100.0
            volatility = 0.0001
            drift = 0.00002
            
        # Generate log returns with drift and volatility
        dt = 1/252/24/60  # Time step in years (1 minute)
        returns = np.random.normal(
            loc=drift*dt, 
            scale=volatility*np.sqrt(dt),
            size=len(dates)
        )
        
        # Calculate price path
        df['close'] = base_price * np.exp(np.cumsum(returns))
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df['close'] * (1 + abs(np.random.normal(0, 0.0003, len(dates))))
        df['low'] = df['close'] * (1 - abs(np.random.normal(0, 0.0003, len(dates))))
        df['volume'] = np.random.lognormal(10, 1, len(dates))
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        return df

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def _load_sentiment_data(self, start: str, end: str):
        """Load sentiment data"""
        logger.info("Loading sentiment data...")
        dates = pd.date_range(start=start, end=end, freq='1min')
        df = pd.DataFrame(index=dates)
        
        # Generate synthetic sentiment scores (-1 to 1)
        df['sentiment'] = np.random.normal(0, 0.2, len(dates))
        df['sentiment'] = df['sentiment'].clip(-1, 1)
        
        self.data['sentiment'] = df
        
    async def _load_fundamentals_data(self, start: str, end: str):
        """Load fundamental data"""
        logger.info("Loading fundamentals data...")
        dates = pd.date_range(start=start, end=end, freq='1min')
        df = pd.DataFrame(index=dates)
        
        # Generate synthetic fundamental metrics
        df['pe_ratio'] = np.random.normal(20, 5, len(dates))
        df['volume'] = np.random.lognormal(10, 1, len(dates))
        df['market_cap'] = np.random.lognormal(25, 2, len(dates))
        
        self.data['fundamentals'] = df

    async def _simulate_trading(self, strategy: BaseStrategy):
        """Simulate trading with realistic conditions"""
        logger.info("Simulating trades...")
        
        # Get common date range across all data
        trading_pairs = self.config['trading']['pairs']
        dates = sorted(set.intersection(*[
            set(data.index) for symbol, data in self.data.items() 
            if symbol in trading_pairs
        ]))
        
        # Simulate trading bar by bar
        for timestamp in dates:
            # Get current market prices
            prices = {
                pair: self.data[pair].loc[timestamp, 'close']
                for pair in trading_pairs if pair in self.data
            }
            
            # Skip if missing prices
            if not prices:
                continue
                
            # Update portfolio NAV history
            nav = self.portfolio.get_nav(prices)
            self.portfolio.nav_history.append((timestamp, nav))
            
            # Get trading signals for each pair
            for pair in trading_pairs:
                if pair not in self.data:
                    continue
                    
                position = self.portfolio.get_position(pair)
                current_price = prices[pair]
                
                # Get strategy signal
                signal = await strategy.analyze(
                    pair=pair,
                    price=current_price,
                    position=position
                )
                
                # Execute trades based on signal
                if signal == 'BUY':
                    size = self._calculate_position_size(pair, current_price)
                    self.portfolio.execute_trade(
                        pair, 'BUY', current_price, size, timestamp
                    )
                elif signal == 'SELL':
                    position = self.portfolio.get_position(pair)
                    if position > 0:
                        self.portfolio.execute_trade(
                            pair, 'SELL', current_price, position, timestamp
                        )

    def _calculate_position_size(self, pair: str, price: float) -> float:
        """Calculate trade size based on position sizing rules"""
        nav = self.portfolio.get_nav({pair: price})
        max_size = self.config['robo']['strategy']['constraints']['max_position_size']
        position_value = nav * max_size
        return position_value / price

    def _calculate_stats(self):
        """Calculate comprehensive performance statistics"""
        if not self.portfolio.nav_history:
            self.stats = {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': float('inf')
            }
            return
            
        nav_df = pd.DataFrame(
            self.portfolio.nav_history, 
            columns=['timestamp', 'nav']
        ).set_index('timestamp')
        
        returns = nav_df['nav'].pct_change().dropna()
        
        # Handle empty returns
        if len(returns) < 2:
            sharpe = 0.0
        else:
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0.0
        
        # Calculate drawdown
        peaks = nav_df['nav'].cummax()
        drawdowns = (nav_df['nav'] - peaks) / peaks
        max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0
        
        # Calculate trade statistics
        trades = self.portfolio.trades
        if trades:
            profits = [t['cost'] for t in trades]
            winning_trades = sum(1 for p in profits if p > 0)
            win_rate = winning_trades / len(profits)
            
            total_profit = sum(p for p in profits if p > 0)
            total_loss = abs(sum(p for p in profits if p < 0))
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = float('inf')
        
        self.stats = {
            'total_trades': len(trades),
            'total_return': (nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0] - 1) if len(nav_df) > 0 else 0.0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
    def _print_results(self):
        """Print detailed backtest results"""
        logger.info("\n=== Backtest Results ===")
        logger.info(f"Total Trades: {self.stats['total_trades']}")
        logger.info(f"Total Return: {self.stats['total_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {self.stats['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {self.stats['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {self.stats['win_rate']*100:.1f}%")
        logger.info(f"Profit Factor: {self.stats['profit_factor']:.2f}")
        logger.info("=====================")

async def main():
    parser = argparse.ArgumentParser(description='FinGPT Backtester')
    parser.add_argument('--config', type=str, default='config/trading.yaml', help='Path to config file')
    parser.add_argument('--start', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-02-07', help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize strategy and backtester
    strategy = TaxAwareStrategy(config=config)
    backtester = Backtester(config)

    # Run backtest
    await backtester.run(strategy, args.start, args.end)

if __name__ == '__main__':
    asyncio.run(main())
