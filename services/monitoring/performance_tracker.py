import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService


class PerformanceTracker(BaseService):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.trades = []
        self.portfolio_values = []
        self.benchmark_data = {}

    async def _setup(self) -> None:
        self.benchmark_symbol = self.config.get("benchmark", "SPY")
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)

    async def add_trade(self, trade: Dict) -> None:
        """Record new trade"""
        self.trades.append({"timestamp": datetime.now(), **trade})

    async def record_trade(self, trade_data: Dict) -> None:
        """Record a completed trade for performance analysis"""
        if not trade_data:
            return
            
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now()
            
        # Calculate P&L if not already included
        if 'pnl' not in trade_data and 'entry_price' in trade_data and 'price' in trade_data:
            quantity = trade_data.get('quantity', 0)
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('price', 0)
            
            # P&L calculation depends on side
            if trade_data.get('side', '').upper() == 'BUY':
                # For a buy, we profit when exit > entry
                trade_data['pnl'] = quantity * (exit_price - entry_price)
            else:
                # For a sell, we profit when entry > exit
                trade_data['pnl'] = quantity * (entry_price - exit_price)
        
        # Add trade to history
        self.trades.append(trade_data)
        
        # Limit history size
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]

    async def calculate_returns(self) -> Dict:
        """Calculate portfolio returns metrics"""
        # Check if we have enough data
        if len(self.portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0
            }
        
        # Ensure we have the necessary columns
        df = pd.DataFrame(self.portfolio_values)
        
        # Check if 'value' column exists, if not try to create it
        if 'value' not in df.columns:
            # Try alternative column names or assign zeros
            if 'portfolio_value' in df.columns:
                df['value'] = df['portfolio_value']
            elif 'total' in df.columns:
                df['value'] = df['total']
            else:
                # Log the error for debugging
                from logging import getLogger
                logger = getLogger(__name__)
                logger.error(f"Portfolio values missing 'value' column. Available columns: {df.columns.tolist()}")
                logger.error(f"First portfolio record: {self.portfolio_values[0] if self.portfolio_values else 'None'}")
                
                # Create default value column to prevent errors
                df['value'] = 10000.0  # Default starting value
        
        returns = {
            "total_return": self._calculate_total_return(df),
            "sharpe_ratio": self._calculate_sharpe_ratio(df),
            "max_drawdown": self._calculate_max_drawdown(df),
            "volatility": self._calculate_volatility(df),
        }
        return returns

    async def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics for the trading system"""
        # Start with basic returns metrics
        metrics = await self.calculate_returns()
        
        # Add trade-based metrics if we have trades
        win_count = 0
        loss_count = 0
        profit_sum = 0.0
        loss_sum = 0.0
        
        # Process trades for metrics
        if self.trades:
            for trade in self.trades:
                # Skip trades without P&L information
                if 'pnl' not in trade:
                    continue
                    
                pnl = trade.get('pnl', 0)
                if pnl > 0:
                    win_count += 1
                    profit_sum += pnl
                elif pnl < 0:
                    loss_count += 1
                    loss_sum += abs(pnl)  # Make positive for calculations
        
        # Calculate win rate
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Calculate profit factor
        metrics['profit_factor'] = profit_sum / loss_sum if loss_sum > 0 else (float('inf') if profit_sum > 0 else 0)
        
        # Calculate expectancy
        if total_trades > 0:
            avg_win = profit_sum / win_count if win_count > 0 else 0
            avg_loss = loss_sum / loss_count if loss_count > 0 else 0
            metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            metrics['avg_trade'] = (profit_sum - loss_sum) / total_trades
        else:
            metrics['expectancy'] = 0
            metrics['avg_trade'] = 0
        
        # Add reward/risk ratio
        if loss_count > 0 and win_count > 0:
            avg_win = profit_sum / win_count
            avg_loss = loss_sum / loss_count
            metrics['reward_risk'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
        else:
            metrics['reward_risk'] = 0
            
        # Add equity curve if we have portfolio history
        if self.portfolio_values:
            metrics['equity_curve'] = [val['value'] for val in self.portfolio_values]
        
        # Add additional forex trading metrics
        if hasattr(self, 'benchmark_data') and self.benchmark_data:
            # Calculate alpha and beta if benchmark data is available
            # (Implementation would depend on how benchmark_data is structured)
            pass
            
        return metrics

    async def record_performance(self, portfolio_summary: Dict) -> None:
        """Record current portfolio performance for historical tracking"""
        if not portfolio_summary:
            return
            
        # Extract total value and timestamp
        timestamp = datetime.now()
        total_value = portfolio_summary.get('total_value', 0)
        
        # If no total value provided but we have positions and cash
        if total_value == 0 and 'positions' in portfolio_summary and 'cash' in portfolio_summary:
            positions_value = sum(pos.get('value', 0) for pos in portfolio_summary['positions'].values())
            total_value = positions_value + portfolio_summary.get('cash', 0)
        
        # Record portfolio value with timestamp
        self.portfolio_values.append({
            'timestamp': timestamp,
            'value': total_value
        })
        
        # Limit history size to prevent memory issues
        if len(self.portfolio_values) > 1000:  # Keep last 1000 data points
            self.portfolio_values = self.portfolio_values[-1000:]

    async def record_pending_order(self, order_data: Dict) -> None:
        """Record pending order data"""
        if not order_data:
            return
            
        # Add timestamp if not present
        if 'timestamp' not in order_data:
            order_data['timestamp'] = datetime.now()
        
        # Record under pending orders
        if not hasattr(self, 'pending_orders'):
            self.pending_orders = []
            
        self.pending_orders.append(order_data)
        
        # Limit pending orders history
        if len(self.pending_orders) > 100:
            self.pending_orders = self.pending_orders[-100:]

    def _calculate_total_return(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 2 or 'value' not in df.columns:
                return 0.0
            return (df["value"].iloc[-1] / df["value"].iloc[0]) - 1
        except Exception as e:
            from logging import getLogger
            logger = getLogger(__name__)
            logger.error(f"Error calculating total return: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 2 or 'value' not in df.columns:
                return 0.0
            returns = df["value"].pct_change().dropna()
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            excess_returns = returns - self.risk_free_rate / 252
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        except Exception as e:
            from logging import getLogger
            logger = getLogger(__name__)
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 2 or 'value' not in df.columns:
                return 0.0
            peak = df["value"].expanding(min_periods=1).max()
            drawdown = (df["value"] - peak) / peak
            return drawdown.min()
        except Exception as e:
            from logging import getLogger
            logger = getLogger(__name__)
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        try:
            if len(df) < 2 or 'value' not in df.columns:
                return 0.0
            returns = df["value"].pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            return returns.std() * np.sqrt(252)
        except Exception as e:
            from logging import getLogger
            logger = getLogger(__name__)
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    async def _cleanup(self) -> None:
        """Clean up any resources used by the performance tracker"""
        # Clear stored data to free memory
        self.trades = []
        self.portfolio_values = []
        self.benchmark_data = {}
        # Log that cleanup completed
        from logging import getLogger
        logger = getLogger(__name__)
        logger.info("Performance tracker cleaned up")