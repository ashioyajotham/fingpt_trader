"""Portfolio management with sentiment-aware allocation"""

from typing import Dict, List, Optional
from ..base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from models.portfolio.risk import MarketRegime

logger = logging.getLogger(__name__)

class PortfolioManager(BaseStrategy):
    """Sentiment and regime-aware portfolio management"""
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        
        # Initialize portfolio dict
        self.portfolio = {
            'balance': config.get('trading', {}).get('initial_balance', 1000.0),
            'positions': {},
            'pnl': 0.0,
            'equity': 0.0
        }
        
        # Portfolio settings
        self.target_volatility = config.get('target_volatility', 0.15)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        self.sentiment_weight = config.get('sentiment_weight', 0.3)
        self.min_sentiment_samples = config.get('min_sentiment_samples', 10)
        self.initial_balance = config.get('trading', {}).get('initial_balance', 1000.0)
        
        # Add missing attributes
        self.max_allocation = config.get('max_position_size', 0.2)
        self.min_allocation = config.get('min_position_size', 0.01)
        self.max_leverage = config.get('max_leverage', 1.0)
        
        # Portfolio tracking
        self.positions = {}  # Current positions
        self.position_values = {}  # Position values in quote currency

    async def _setup(self) -> None:
        """Initialize portfolio state"""
        try:
            # Initialize positions
            self.positions = {
                'BTCUSDT': {'size': 0.1, 'entry_price': 0},
                'ETHUSDT': {'size': 1.0, 'entry_price': 0},
            }
            # Update portfolio dict
            self.portfolio['positions'] = self.positions
            self.portfolio['balance'] = self.initial_balance
            self.portfolio['equity'] = self.initial_balance
            
            logger.info(f"Portfolio initialized with balance: {self.initial_balance}")
            
        except Exception as e:
            logger.error(f"Portfolio initialization failed: {e}")
            raise

    async def _cleanup(self) -> None:
        """Cleanup portfolio resources"""
        try:
            self.positions.clear()
            self.portfolio.clear()
            logger.info("Portfolio cleaned up successfully")
        except Exception as e:
            logger.error(f"Portfolio cleanup failed: {e}")
            raise

    async def initialize(self) -> None:
        """Public initialize method"""
        await self._setup()

    async def generate_signals(self, market_data: Dict) -> List[Dict]:
        """Generate trading signals - public async wrapper"""
        return await self._generate_base_signals(market_data)

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

    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate current portfolio allocations as percentages"""
        total_value = sum(abs(val) for val in self.position_values.values())
        
        if total_value == 0:
            return {symbol: 0.0 for symbol in self.active_pairs}
            
        return {
            symbol: abs(self.position_values.get(symbol, 0.0)) / total_value
            for symbol in self.positions.keys()
        }

    def _calculate_risk_metrics(self, market_data: Dict) -> Dict:
        """Calculate risk metrics for portfolio management"""
        metrics = {}
        
        for symbol in self.active_pairs:
            # Skip if no market data
            if not market_data.get('market', {}).get('candles', {}).get(symbol):
                continue
                
            # Get price data
            candles = market_data['market']['candles'][symbol]
            
            # Calculate metrics if we have enough data
            if len(candles) >= 20:  # Minimum sample size
                prices = pd.Series([float(c[4]) for c in candles])  # Close prices
                returns = prices.pct_change().dropna()
                
                metrics[symbol] = {
                    'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
                    'returns': returns.mean() * 252,  # Annualized returns
                    'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_drawdown(prices),
                    'current_price': float(candles[-1][4])
                }
            else:
                # Use default values if not enough data
                metrics[symbol] = {
                    'volatility': 1.0,  # High volatility assumption
                    'returns': 0.0,
                    'sharpe': 0.0,
                    'max_drawdown': 0.0,
                    'current_price': float(candles[-1][4]) if candles else 0.0
                }
                
        return metrics

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

    def _calculate_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative PnL"""
        rolling_max = cumulative_pnl.expanding().max()
        drawdowns = cumulative_pnl - rolling_max
        return abs(drawdowns.min())

    async def analyze(self, pair: str, current_price: float = None, price: float = None,
                     position: float = 0, position_size: float = None,
                     holding_period: int = None, unrealized_pnl: float = None) -> Optional[str]:
        """Analyze portfolio state and generate rebalancing signals"""
        # Get current allocation
        current_alloc = self._calculate_allocations()
        current_pos = current_alloc.get(pair, 0.0)
        
        # Get target allocation
        market_data = {
            'price': current_price or price,
            'position': position_size or position,
            'pair': pair
        }
        target_alloc = await self._calculate_target_allocations({'market': market_data})
        target_pos = target_alloc.get(pair, 0.0)
        
        # Generate signal based on allocation difference
        if abs(target_pos - current_pos) > self.rebalance_threshold:
            return 'BUY' if target_pos > current_pos else 'SELL'
        
        return None

    async def on_trade(self, trade_data: Dict) -> None:
        """Handle trade execution updates"""
        symbol = trade_data.get('symbol')
        if not symbol:
            return
            
        # Update position tracking
        if symbol not in self.trades_history:
            self.trades_history[symbol] = []
            
        self.trades_history[symbol].append({
            'timestamp': datetime.now(),
            'price': trade_data.get('price', 0),
            'quantity': trade_data.get('quantity', 0),
            'side': trade_data.get('side', 'UNKNOWN')
        })
        
        # Update performance metrics
        self._update_performance_metrics()

    async def validate(self, **kwargs) -> bool:
        """Validate portfolio strategy configuration"""
        try:
            # Validate allocation limits
            if not (0 < self.min_allocation <= self.max_allocation <= 1.0):
                return False
                
            # Validate leverage constraints
            if not (1.0 <= self.max_leverage <= 3.0):
                return False
                
            # Validate volatility target
            if not (0.05 <= self.target_volatility <= 0.5):
                return False
                
            # Validate rebalance threshold
            if not (0.01 <= self.rebalance_threshold <= 0.2):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Portfolio validation failed: {str(e)}")
            return False

    async def _analyze_market_state(self) -> Dict:
        """Analyze current market state"""
        try:
            # Get market data
            recent_candles = self.market_state.get('candles', {})
            recent_trades = self.market_state.get('trades', {})
            orderbooks = self.market_state.get('orderbooks', {})
            
            # Calculate market state indicators
            volatility = self._calculate_volatility()
            liquidity = self._calculate_liquidity()
            
            # Calculate total volume - fix for float iteration error
            total_volume = 0
            for symbol, trades in recent_trades.items():
                if isinstance(trades, list):
                    total_volume += sum(float(trade.get('volume', 0)) for trade in trades)
            
            # Detect market regime
            regime_data = {
                'volatility': volatility,
                'volume': total_volume,  # Now using properly calculated volume
                'spread': self._calculate_average_spread(orderbooks),
                'price_impact': self._estimate_price_impact(orderbooks)
            }
            
            current_regime = self.regime_detector.detect_regime(regime_data)
            
            return {
                'regime': current_regime,
                'volatility': volatility,
                'liquidity': liquidity,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market state: {e}")
            # Return default safe state
            return {
                'regime': MarketRegime.HIGH_VOL,  # Conservative default
                'volatility': 1.0,
                'liquidity': 0.0,
                'timestamp': datetime.now()
            }

    def _calculate_average_spread(self, orderbooks: Dict) -> float:
        """Calculate average spread across pairs"""
        spreads = []
        for symbol, ob in orderbooks.items():
            if ob.get('asks') and ob.get('bids'):
                best_ask = float(ob['asks'][0][0])
                best_bid = float(ob['bids'][0][0])
                spread = (best_ask - best_bid) / best_bid
                spreads.append(spread)
        return np.mean(spreads) if spreads else 1.0

    def _estimate_price_impact(self, orderbooks: Dict) -> float:
        """Estimate market impact for standard trade size"""
        impacts = []
        for symbol, ob in orderbooks.items():
            if ob.get('asks') and ob.get('bids'):
                # Calculate impact for 1% of order book depth
                depth = sum(float(level[1]) for level in ob['bids'][:5])
                standard_size = depth * 0.01
                impact = self._calculate_impact(standard_size, ob['asks'])
                impacts.append(impact)
        return np.mean(impacts) if impacts else 1.0

    def _calculate_impact(self, size: float, asks: List) -> float:
        """Calculate price impact for given size"""
        remaining_size = size
        weighted_price = 0
        initial_price = float(asks[0][0])
        
        for price, quantity in asks:
            price = float(price)
            quantity = float(quantity)
            if remaining_size <= 0:
                break
            executed = min(remaining_size, quantity)
            weighted_price += price * executed
            remaining_size -= executed
            
        return (weighted_price / size - initial_price) / initial_price if size > 0 else 0