from ..base_service import BaseService
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
import logging
from pathlib import Path

class PerformanceTracker(BaseService):
    def _validate_config(self) -> None:
        required = ['data_path', 'report_interval']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def initialize(self) -> None:
        self.trades = []
        self.positions = {}
        self.daily_stats = {}
        self.metrics = {}
        self.data_path = Path(self.config['data_path'])
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._load_historical_data()

    async def shutdown(self) -> None:
        """Save performance data before shutdown"""
        self._save_historical_data()

    async def track_trade(self, trade: Dict) -> None:
        """Record a new trade"""
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        await self._update_metrics()
        
        # Save after significant events
        if len(self.trades) % 10 == 0:  # Every 10 trades
            self._save_historical_data()

    async def update_position(self, symbol: str, position: Dict) -> None:
        """Update current position information"""
        self.positions[symbol] = {
            **position,
            'last_updated': datetime.now().isoformat()
        }
        await self._update_metrics()

    async def _update_metrics(self) -> None:
        """Calculate performance metrics"""
        if not self.trades:
            return

        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Daily statistics
        daily = df.set_index('timestamp').resample('D').agg({
            'pnl': 'sum',
            'quantity': 'sum',
            'commission': 'sum'
        }).fillna(0)

        # Calculate key metrics
        self.metrics = {
            'total_pnl': df['pnl'].sum(),
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean(),
            'avg_win': df[df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean(),
            'sharpe_ratio': self._calculate_sharpe(daily['pnl']),
            'max_drawdown': self._calculate_drawdown(daily['pnl'].cumsum()),
            'last_updated': datetime.now().isoformat()
        }

        # Update daily stats
        self.daily_stats = daily.to_dict(orient='index')

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def _calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(drawdowns.min())

    def get_current_metrics(self) -> Dict:
        """Get latest performance metrics"""
        return self.metrics

    async def generate_report(self, start_date: Optional[datetime] = None) -> Dict:
        """Generate performance report"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)

        df = pd.DataFrame(self.trades)
        if df.empty:
            return {"error": "No trading data available"}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= start_date]

        return {
            'summary_metrics': self.metrics,
            'daily_stats': {
                k: v for k, v in self.daily_stats.items()
                if pd.to_datetime(k) >= start_date
            },
            'position_summary': self.positions,
            'report_generated': datetime.now().isoformat()
        }

    def _save_historical_data(self) -> None:
        """Save performance data to disk"""
        try:
            data = {
                'trades': self.trades,
                'metrics': self.metrics,
                'daily_stats': self.daily_stats
            }
            with open(self.data_path / 'performance_history.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")

    def _load_historical_data(self) -> None:
        """Load historical performance data"""
        try:
            history_file = self.data_path / 'performance_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.metrics = data.get('metrics', {})
                    self.daily_stats = data.get('daily_stats', {})
        except Exception as e:
            self.logger.error(f"Error loading performance data: {str(e)}")
