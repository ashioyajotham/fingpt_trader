from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum
from typing import TypeVar, Protocol


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
    def __init__(self, max_drawdown: float = 0.1, var_limit: float = 0.02):
        self.max_drawdown = max_drawdown
        self.var_limit = var_limit
        self.historical_values = []

    def calculate_risk_metrics(self, portfolio: Dict) -> Dict:
        """Calculate portfolio risk metrics"""
        positions = portfolio.get('positions', np.array([]))
        values = portfolio.get('values', np.array([]))
        
        if len(values) == 0:
            return {'var': 0.0, 'current_drawdown': 0.0}
        
        # Calculate Value at Risk (VaR)
        var = self._calculate_var(values)
        
        # Calculate current drawdown
        self.historical_values.append(np.sum(values))
        current_drawdown = self._calculate_drawdown()
        
        return {
            'var': var,
            'current_drawdown': current_drawdown,
            'position_concentration': self._calculate_concentration(values),
            'total_exposure': self._calculate_exposure(values)
        }

    def _calculate_var(self, values: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(self.historical_values) < 2:
            return 0.0
        
        returns = np.diff(self.historical_values) / self.historical_values[:-1]
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var * np.sum(values))

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if len(self.historical_values) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(self.historical_values)
        drawdown = (peak - self.historical_values) / peak
        return float(drawdown[-1])

    def _calculate_concentration(self, values: np.ndarray) -> float:
        """Calculate Herfindahl index for position concentration"""
        total = np.sum(values)
        if total == 0:
            return 0.0
        weights = values / total
        return float(np.sum(weights ** 2))

    def _calculate_exposure(self, values: np.ndarray) -> float:
        """Calculate total market exposure"""
        return float(np.sum(np.abs(values)))


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
    def __init__(self):
        self.indicators = {
            'volatility': VolatilityIndicator(),
            'liquidity': LiquidityMetrics(),
            'correlation': CorrelationMatrix(),
            'sentiment': SentimentIndicator()
        }
        
    def detect_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime using multiple indicators"""
        signals = {}
        
        # Get signals from all indicators
        for name, indicator in self.indicators.items():
            try:
                signals[name] = indicator.get_signal(market_data.get(name, {}))
            except Exception as e:
                signals[name] = 0.0
                
        # Regime detection logic
        if signals['volatility'] > 2.0:
            return MarketRegime.HIGH_VOL
        elif signals['liquidity'] > 0.05:
            return MarketRegime.LOW_LIQUIDITY
        elif signals['correlation'] > 0.8:
            return MarketRegime.STRESS
        elif sum(signals.values()) / len(signals) > 1.5:
            return MarketRegime.CRISIS
            
        return MarketRegime.NORMAL


class CircuitBreaker:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = {
            'volatility': thresholds.get('volatility', 3.0),
            'volume': thresholds.get('volume', 5.0),
            'spread': thresholds.get('spread', 0.05),
            'imbalance': thresholds.get('imbalance', 0.7)
        }
        
    def check_conditions(self, market_data: Dict) -> bool:
        """Check market conditions against circuit breaker rules"""
        try:
            # Check volatility threshold
            if market_data.get('volatility', 0) > self.thresholds['volatility']:
                return True
                
            # Check volume spike
            if market_data.get('volume_change', 0) > self.thresholds['volume']:
                return True
                
            # Check bid-ask spread
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            if bid and ask:
                spread = (ask - bid) / ((ask + bid) / 2)
                if spread > self.thresholds['spread']:
                    return True
                    
            # Check order book imbalance
            bids_volume = sum(bid[1] for bid in market_data.get('bids', []))
            asks_volume = sum(ask[1] for ask in market_data.get('asks', []))
            total_volume = bids_volume + asks_volume
            if total_volume > 0:
                imbalance = abs(bids_volume - asks_volume) / total_volume
                if imbalance > self.thresholds['imbalance']:
                    return True
                    
            return False
            
        except Exception as e:
            # Log error and trigger circuit breaker for safety
            return True
