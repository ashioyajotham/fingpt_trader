
from typing import Dict, List, Optional
from ..base_strategy import BaseStrategy
import numpy as np

class PortfolioStrategy(BaseStrategy):
    """Portfolio management strategy with risk-aware allocation"""
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        
        # Portfolio constraints
        self.max_allocation = config.get('max_position_size', 0.2)
        self.min_allocation = config.get('min_position_size', 0.01)
        self.max_leverage = config.get('max_leverage', 1.0)
        
        # Risk parameters
        self.target_volatility = config.get('target_volatility', 0.15)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        
        # Tax-aware trading
        self.tax_aware = config.get('tax_aware', True)
        self.tax_rate = profile.get('tax_rate', 0.25) if profile else 0.25
        self.min_holding_days = config.get('min_holding_period', 30)

    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        """Generate portfolio rebalancing signals"""
        signals = []
        
        # Get current allocations
        current_alloc = self._calculate_allocations()
        
        # Calculate target allocations
        target_alloc = self._optimize_portfolio(market_data)
        
        # Generate rebalancing signals
        for symbol, target in target_alloc.items():
            current = current_alloc.get(symbol, 0.0)
            if abs(target - current) > self.rebalance_threshold:
                signals.append({
                    'symbol': symbol,
                    'direction': 1 if target > current else -1,
                    'size': abs(target - current) * self.initial_balance,
                    'strength': min(abs(target - current) / self.rebalance_threshold, 1.0),
                    'type': 'rebalance'
                })
                
        return signals

    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate current portfolio allocations"""
        total_value = sum(abs(pos) for pos in self.positions.values())
        if total_value == 0:
            return {symbol: 0.0 for symbol in self.active_pairs}
            
        return {
            symbol: abs(pos)/total_value 
            for symbol, pos in self.positions.items()
        }

    def _optimize_portfolio(self, market_data: Dict) -> Dict[str, float]:
        """Optimize portfolio allocations based on risk/return"""
        # Calculate volatility and correlations
        vol_data = self._calculate_risk_metrics(market_data)
        
        # Risk-based allocation
        allocations = {}
        for symbol in self.active_pairs:
            vol = vol_data.get(symbol, {'volatility': 1.0})['volatility']
            allocations[symbol] = self.target_volatility / (vol + 1e-10)
            
        # Normalize allocations
        total = sum(allocations.values())
        if total > 0:
            allocations = {
                s: min(alloc/total, self.max_allocation) 
                for s, alloc in allocations.items()
            }
            
        return allocations

    def _calculate_risk_metrics(self, market_data: Dict) -> Dict:
        """Calculate volatility and correlation metrics"""
        metrics = {}
        for symbol in self.active_pairs:
            if symbol not in market_data.get('candles', {}):
                continue
                
            candles = market_data['candles'][symbol]
            closes = [float(c[4]) for c in candles[-20:]]
            
            if len(closes) > 1:
                returns = np.diff(np.log(closes))
                metrics[symbol] = {
                    'volatility': np.std(returns) * np.sqrt(252),
                    'returns': returns.mean() * 252
                }
                
        return metrics

    async def process_market_data(self, market_data: Dict) -> Dict:
        """Process market data for portfolio updates"""
        analysis = await super().process_market_data(market_data)
        
        # Add portfolio-specific metrics
        analysis.update({
            'allocations': self._calculate_allocations(),
            'risk_metrics': self._calculate_risk_metrics(market_data)
        })
        
        return analysis
