from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum
from typing import TypeVar, Protocol
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    volatility: float
    var: float  # Value at Risk
    es: float  # Expected Shortfall
    beta: float
    tracking_error: Optional[float] = None


class RiskAnalyzer:
    def __init__(self, config: Dict):
        self.confidence_level = config.get("confidence_level", 0.95)
        self.lookback_window = config.get("lookback_window", 252)
        self.use_ewma = config.get("use_ewma", True)

    def calculate_portfolio_risk(
        self,
        returns: pd.Series,
        weights: Dict[str, float],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Calculate portfolio returns if weights provided
        if isinstance(returns, pd.DataFrame):
            port_returns = (returns * pd.Series(weights)).sum(axis=1)
        else:
            port_returns = returns

        # Calculate volatility
        if self.use_ewma:
            vol = self._calculate_ewma_volatility(port_returns)
        else:
            vol = port_returns.std() * np.sqrt(252)

        # Calculate VaR and ES
        var = self._calculate_var(port_returns)
        es = self._calculate_expected_shortfall(port_returns)

        # Calculate beta if benchmark provided
        beta = (
            self._calculate_beta(port_returns, benchmark_returns)
            if benchmark_returns is not None
            else None
        )

        # Calculate tracking error if benchmark provided
        tracking_error = (
            self._calculate_tracking_error(port_returns, benchmark_returns)
            if benchmark_returns is not None
            else None
        )

        return RiskMetrics(
            volatility=vol,
            var=var,
            es=es,
            beta=beta if beta is not None else 0,
            tracking_error=tracking_error,
        )

    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """Calculate EWMA volatility"""
        lambda_param = 0.94
        return np.sqrt(252) * np.sqrt(
            returns.ewm(alpha=1 - lambda_param).var().iloc[-1]
        )

    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk"""
        return -np.percentile(returns, 100 * (1 - self.confidence_level))

    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self._calculate_var(returns)
        return -returns[returns <= -var].mean()

    def _calculate_beta(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Calculate portfolio beta"""
        covar = returns.cov(benchmark_returns)
        benchmark_var = benchmark_returns.var()
        return covar / benchmark_var if benchmark_var != 0 else 0

    def _calculate_tracking_error(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error"""
        return (returns - benchmark_returns).std() * np.sqrt(252)


class RiskManager:
    """Portfolio risk management"""
    
    def __init__(self, max_drawdown: float = 0.1, var_limit: float = 0.02):
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.historical_values = []
        self._initialize_state()

    def _initialize_state(self):
        """Initialize risk tracking state"""
        self.peak_value = 0.0
        self.current_value = 0.0
        self.position_values = {}
        self.historical_values = []

    def calculate_risk_metrics(self, portfolio: Dict) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            # Extract values from portfolio
            values = portfolio.get('values', {})
            total_value = sum(float(v) for v in values.values())
            
            # Update historical tracking
            self.historical_values.append(total_value)
            if len(self.historical_values) > 1000:  # Limit history size
                self.historical_values = self.historical_values[-1000:]
            
            # Update peak tracking
            self.peak_value = max(self.peak_value, total_value)
            self.current_value = total_value
            
            # Calculate metrics
            position_values = {k: float(v) for k, v in values.items()}
            
            return {
                'max_drawdown': self._calculate_drawdown(),
                'var': self._calculate_var(),
                'position_concentration': self._calculate_concentration(position_values),
                'total_exposure': self._calculate_exposure(position_values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'max_drawdown': 0.0,
                'var': 0.0,
                'position_concentration': 0.0,
                'total_exposure': 0.0
            }

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.historical_values:
            return 0.0
            
        values = np.array(self.historical_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return float(np.max(drawdown))

    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(self.historical_values) < 2:
            return 0.0
            
        returns = np.diff(self.historical_values) / self.historical_values[:-1]
        return float(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_concentration(self, values: Dict[str, float]) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        if not values:
            return 0.0
            
        total = sum(values.values())
        if total == 0:
            return 0.0
            
        weights = np.array([v/total for v in values.values()])
        return float(np.sum(weights * weights))

    def _calculate_exposure(self, values: Dict[str, float]) -> float:
        """Calculate total portfolio exposure"""
        if not values:
            return 0.0
            
        total = sum(values.values())
        if total == 0:
            return 0.0
            
        exposure = sum(abs(v) for v in values.values())
        return float(exposure / total)

    def check_risk_limits(self, metrics: Dict) -> bool:
        """Check if risk metrics are within limits"""
        if metrics['max_drawdown'] > self.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {metrics['max_drawdown']:.2%}")
            return False
            
        if abs(metrics['var']) > self.var_limit:
            logger.warning(f"VaR limit exceeded: {metrics['var']:.2%}")
            return False
            
        return True


class MarketRegime(Enum):
    NORMAL = "normal"
    STRESS = "stress"
    CRISIS = "crisis"
    HIGH_VOL = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"


class MarketIndicator(Protocol):
    def get_signal(self) -> float:
        ...


class VolatilityIndicator:
    def __init__(self, window: int = 20):
        self.window = window
        self.threshold = 2.0  # Standard deviations

    def get_signal(self, returns: pd.Series) -> float:
        vol = returns.rolling(self.window).std() * np.sqrt(252)
        return float(vol.iloc[-1] if not vol.empty else 0.0)


class LiquidityMetrics:
    def __init__(self, spread_threshold: float = 0.02):
        self.spread_threshold = spread_threshold
    
    def get_signal(self, market_data: Dict) -> float:
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        if bid == 0 or ask == 0:
            return 0.0
        return (ask - bid) / ((ask + bid) / 2)


class CorrelationMatrix:
    def __init__(self, window: int = 30):
        self.window = window
        
    def get_signal(self, returns: pd.DataFrame) -> float:
        if len(returns) < self.window:
            return 0.0
        corr = returns.rolling(self.window).corr()
        return float(corr.mean().mean())


class SentimentIndicator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def get_signal(self, sentiment_scores: List[float]) -> float:
        if not sentiment_scores:
            return 0.0
        return float(np.mean(sentiment_scores))


class MarketRegimeDetector:
    """Detects market regime based on multiple indicators"""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Initialize with default thresholds
        self.thresholds = {
            'volatility': {
                'high': config.get('volatility_high', 0.03),    # 3% daily volatility
                'stress': config.get('volatility_stress', 0.05),  # 5% daily volatility
                'crisis': config.get('volatility_crisis', 0.07)   # 7% daily volatility
            },
            'volume': {
                'low': config.get('volume_low', 0.5),     # 50% below average
                'crisis': config.get('volume_crisis', 0.2)    # 80% below average
            },
            'spread': {
                'high': config.get('spread_high', 0.002),   # 20bps spread
                'stress': config.get('spread_stress', 0.005)  # 50bps spread
            },
            'price_impact': {
                'high': config.get('price_impact_high', 0.001),   # 10bps for standard size
                'stress': config.get('price_impact_stress', 0.002)  # 20bps for standard size
            }
        }
        
        # Regime history
        self.history = []
        self.max_history = config.get('max_history', 100)

    def detect_regime(self, data: Dict) -> MarketRegime:
        """Detect current market regime from market data"""
        try:
            vol = data.get('volatility', 0)
            volume = data.get('volume', 0)
            spread = data.get('spread', 0)
            impact = data.get('price_impact', 0)
            
            # Crisis conditions
            if (vol > self.thresholds['volatility']['crisis'] or
                volume < self.thresholds['volume']['crisis']):
                regime = MarketRegime.CRISIS
                
            # Stress conditions
            elif (vol > self.thresholds['volatility']['stress'] or
                  spread > self.thresholds['spread']['stress'] or
                  impact > self.thresholds['price_impact']['stress']):
                regime = MarketRegime.STRESS
                
            # High volatility conditions
            elif vol > self.thresholds['volatility']['high']:
                regime = MarketRegime.HIGH_VOL
                
            # Low liquidity conditions
            elif volume < self.thresholds['volume']['low']:
                regime = MarketRegime.LOW_LIQUIDITY
                
            # Normal conditions
            else:
                regime = MarketRegime.NORMAL
                
            # Update history
            self._update_history(regime)
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.HIGH_VOL  # Conservative default
            
    def _update_history(self, regime: MarketRegime) -> None:
        """Update regime history"""
        self.history.append({
            'regime': regime,
            'timestamp': datetime.now()
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)


class CircuitBreaker:
    """Trading circuit breaker implementation"""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or {
            'price_change': 0.1,    # 10% price change
            'volume_spike': 5.0,     # 5x normal volume
            'spread_widening': 0.01  # 100bps spread
        }
        self.triggered = False
        self.trigger_time = None
        self.cooldown_minutes = 30

    def check_conditions(self, market_data: Dict) -> bool:
        """Check if circuit breaker conditions are met"""
        try:
            # Skip if in cooldown
            if self._in_cooldown():
                return self.triggered
                
            # Check conditions
            price_change = abs(market_data.get('price_change', 0))
            volume_ratio = market_data.get('volume_ratio', 1.0)
            spread = market_data.get('spread', 0)
            
            # Trigger breaker if any threshold exceeded
            if (price_change > self.thresholds['price_change'] or
                volume_ratio > self.thresholds['volume_spike'] or
                spread > self.thresholds['spread_widening']):
                
                self._trigger()
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in circuit breaker: {e}")
            return True  # Conservative default
            
    def _trigger(self) -> None:
        """Trigger the circuit breaker"""
        self.triggered = True
        self.trigger_time = datetime.now()
        logger.warning("Circuit breaker triggered")

    def _in_cooldown(self) -> bool:
        """Check if circuit breaker is in cooldown period"""
        if not self.triggered or not self.trigger_time:
            return False
            
        elapsed = (datetime.now() - self.trigger_time).total_seconds() / 60
        if elapsed > self.cooldown_minutes:
            self.triggered = False
            self.trigger_time = None
            return False
            
        return True

    def reset(self) -> None:
        """Reset circuit breaker state"""
        self.triggered = False
        self.trigger_time = None
