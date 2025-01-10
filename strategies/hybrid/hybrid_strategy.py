from typing import Dict, List, Optional
import numpy as np

import sys
from pathlib import Path
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from strategies.base_strategy import BaseStrategy
from strategies.sentiment.sentiment_strategy import SentimentStrategy
from strategies.inefficiency.inefficiency_strategy import InefficiencyStrategy

class HybridStrategy(BaseStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Initialize sub-strategies
        self.strategies = {
            'sentiment': SentimentStrategy(config.get('sentiment_config')),
            'inefficiency': InefficiencyStrategy(config.get('inefficiency_config'))
        }
        
        # Strategy weights
        self.weights = config.get('strategy_weights', {
            'sentiment': 0.5,
            'inefficiency': 0.5
        })
        
        # Correlation matrix
        self.correlation_matrix = np.eye(len(self.strategies))
        self.signal_history = []

    async def generate_signals(self) -> List[Dict]:
        """Generate combined signals from all strategies"""
        all_signals = []
        
        # Collect signals from each strategy
        for name, strategy in self.strategies.items():
            signals = await strategy.generate_signals()
            for signal in signals:
                signal['strategy'] = name
                signal['weight'] = self.weights[name]
            all_signals.extend(signals)
            
        # Combine signals
        combined = await self._combine_signals(all_signals)
        self.signal_history.append(combined)
        
        return combined

    async def _combine_signals(self, signals: List[Dict]) -> List[Dict]:
        """Combine signals using weights and correlations"""
        if not signals:
            return []
            
        combined = {}
        for signal in signals:
            symbol = signal.get('symbol')
            if symbol not in combined:
                combined[symbol] = {
                    'symbol': symbol,
                    'direction': 0,
                    'strength': 0,
                    'strategies': []
                }
                
            weight = signal['weight']
            combined[symbol]['direction'] += signal['direction'] * weight
            combined[symbol]['strength'] += signal['strength'] * weight
            combined[symbol]['strategies'].append(signal['strategy'])
            
        return list(combined.values())

    async def _update_weights(self) -> None:
        """Update strategy weights based on performance"""
        if len(self.signal_history) < 10:
            return
            
        # Calculate strategy returns
        returns = self._calculate_strategy_returns()
        
        # Update correlation matrix
        self.correlation_matrix = np.corrcoef(returns)
        
        # Adjust weights using return/risk ratio
        total_return = np.sum(returns, axis=1)
        total_risk = np.sqrt(np.diag(returns @ self.correlation_matrix @ returns.T))
        
        new_weights = (total_return / total_risk)
        new_weights = new_weights / np.sum(new_weights)
        
        for i, strategy in enumerate(self.strategies):
            self.weights[strategy] = new_weights[i]