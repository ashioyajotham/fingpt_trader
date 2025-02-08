"""Portfolio management with sentiment-aware allocation"""

from typing import Dict, List, Optional
from ..base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from datetime import datetime

class PortfolioManager(BaseStrategy):
    """Sentiment and regime-aware portfolio management"""
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        
        # Portfolio settings
        self.target_volatility = config.get('target_volatility', 0.15)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        self.sentiment_weight = config.get('sentiment_weight', 0.3)
        self.min_sentiment_samples = config.get('min_sentiment_samples', 10)

    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        """Generate portfolio rebalancing signals"""
        signals = []
        
        # Current allocations and state
        current_alloc = self._calculate_allocations()
        target_alloc = await self._calculate_target_allocations(market_data)
        
        # Process sentiment data
        sentiment_scores = market_data.get('sentiment', {})
        market_state = await self._analyze_market_state()
        
        for symbol in self.active_pairs:
            # Skip if no sentiment data
            if symbol not in sentiment_scores:
                continue
                
            current = current_alloc.get(symbol, 0.0)
            target = target_alloc.get(symbol, 0.0)
            sentiment = sentiment_scores.get(symbol, 0.0)
            
            # Adjust target based on sentiment
            sentiment_adj = target * (1 + sentiment * self.sentiment_weight)
            final_target = min(sentiment_adj, self.max_position)
            
            # Generate signal if difference exceeds threshold
            if abs(final_target - current) > self.rebalance_threshold:
                signals.append({
                    'symbol': symbol,
                    'direction': 1 if final_target > current else -1,
                    'size': abs(final_target - current) * self.initial_balance,
                    'strength': min(abs(final_target - current) / self.rebalance_threshold, 1.0),
                    'type': 'rebalance',
                    'regime': market_state['regime']
                })
                
        return signals

    async def _calculate_target_allocations(self, market_data: Dict) -> Dict[str, float]:
        """Calculate target allocations using risk and sentiment"""
        # Get risk metrics
        risk_data = self._calculate_risk_metrics(market_data)
        
        # Calculate inverse volatility weights
        weights = {}
        total_weight = 0
        
        for symbol in self.active_pairs:
            if symbol not in risk_data:
                continue
                
            vol = risk_data[symbol]['volatility']
            if vol > 0:
                weight = self.target_volatility / vol
                weights[symbol] = weight
                total_weight += weight
                
        # Normalize weights
        if total_weight > 0:
            return {
                symbol: min(weight/total_weight, self.max_position)
                for symbol, weight in weights.items()
            }
            
        return {symbol: 0.0 for symbol in self.active_pairs}

    def _update_performance_metrics(self) -> None:
        """Update portfolio performance metrics"""
        if not self.trades_history:
            return
            
        df = pd.DataFrame(self.trades_history)
        
        self.performance_metrics = {
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean(),
            'avg_return': df['pnl'].mean(),
            'sharpe': df['pnl'].mean() / df['pnl'].std() if len(df) > 1 else 0,
            'max_drawdown': self._calculate_drawdown(df['cumulative_pnl'])
        }

    # ...rest of implementation...
