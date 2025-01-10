from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import timedelta

@dataclass
class SentimentConfig:
    # Signal Generation
    threshold: float = 0.3
    min_samples: int = 5
    time_decay: float = 0.95
    lookback_window: timedelta = timedelta(hours=24)
    
    # Trading Parameters
    position_size: float = 1000.0
    max_position: float = 10000.0
    stop_loss: float = 0.02
    take_profit: float = 0.05
    
    # Risk Management
    max_positions: int = 5
    max_concentration: float = 0.2
    daily_drawdown_limit: float = 0.03
    
    # Model Settings
    sentiment_weights: Dict[str, float] = field(default_factory=lambda: {
        'vader': 0.7,
        'textblob': 0.3
    })
    
    # Backtesting
    backtest_start: str = '2024-01-01'
    backtest_end: str = '2024-12-31'
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.threshold <= 1.0
        assert self.min_samples > 0
        assert 0 < self.time_decay <= 1.0
        assert self.position_size > 0
        assert self.max_position >= self.position_size
        assert sum(self.sentiment_weights.values()) == 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'threshold': self.threshold,
            'min_samples': self.min_samples,
            'time_decay': self.time_decay,
            'lookback_window': str(self.lookback_window),
            'position_size': self.position_size,
            'max_position': self.max_position,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_positions': self.max_positions,
            'max_concentration': self.max_concentration,
            'daily_drawdown_limit': self.daily_drawdown_limit,
            'sentiment_weights': self.sentiment_weights,
            'backtest_start': self.backtest_start,
            'backtest_end': self.backtest_end
        }