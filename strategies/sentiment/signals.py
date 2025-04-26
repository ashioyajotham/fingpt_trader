from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate trading signals from sentiment and market data"""
    
    def __init__(self, config: Dict, trading_system=None):
        """Initialize with strategy configuration"""
        self.trading_system = trading_system
        self._validate_config(config)
        self.config = config
        
        # Get threshold values directly from config without fallbacks
        self.detection_threshold = self.config['detection_threshold']
        
        # Get execution threshold from trading.yaml
        self.execution_threshold = self.trading_system.get_config('trading.execution.signal_threshold')
        
        # Get performance thresholds
        self.confidence_threshold = self.config['confidence_threshold']
        self.accuracy_threshold = self.config['accuracy_threshold']
        
        # Get signal weights from config
        self.impact_weights = self.config['impact_weights']
        
        # Initialize tracking
        self.signals = []
        self.min_samples = self.config['min_samples']
        self.performance = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0
        }

    def generate(self, 
                sentiment_score: float,
                market_data: Dict,
                timestamp: Optional[datetime] = None) -> Dict:
        """Generate trading signal"""
        # Validate inputs
        if not self._validate_inputs(sentiment_score, market_data):
            return self._neutral_signal()
            
        # Calculate signal components
        sentiment_signal = self._process_sentiment(sentiment_score)
        market_signal = self._process_market_data(market_data)
        
        # Combine signals
        signal = self._combine_signals(sentiment_signal, market_signal)
        
        # Add metadata
        signal['timestamp'] = timestamp or datetime.now()
        signal['market_data'] = market_data
        
        # Track signal
        self.signals.append(signal)
        
        return signal

    def _process_sentiment(self, score: float) -> Dict:
        """Process sentiment score into signal component"""
        if abs(score) < self.detection_threshold:
            return {'value': 0, 'confidence': 0}
            
        return {
            'value': np.sign(score) * min(abs(score), 1.0),
            'confidence': min(abs(score) * 1.5, 1.0)  # Scale up confidence
        }

    def _process_market_data(self, data: Dict) -> Dict:
        """Process market data into signal component"""
        # Extract relevant metrics
        volume = data.get('volume', 0)
        volatility = data.get('volatility', 0)
        trend = data.get('trend', 0)
        
        # Calculate market signal
        market_score = (
            0.4 * np.sign(trend) * min(abs(trend), 1.0) +
            0.3 * (volume - 1.0) +  # Volume relative to average
            0.3 * min(volatility, 2.0) / 2.0
        )
        
        return {
            'value': market_score,
            'confidence': min(
                (volume + abs(trend) + volatility) / 3,
                1.0
            )
        }

    def _combine_signals(self, 
                        sentiment: Dict,
                        market: Dict) -> Dict:
        """Combine signal components into final signal"""
        # Weighted combination
        weights = self.impact_weights
        signal_value = (
            weights['sentiment'] * sentiment['value'] +
            weights['volume'] * market['value']
        )
        
        # Combined confidence
        confidence = (
            weights['sentiment'] * sentiment['confidence'] +
            weights['volume'] * market['confidence']
        )
        
        return {
            'direction': np.sign(signal_value),
            'strength': abs(signal_value),
            'confidence': confidence,
            'sentiment': sentiment,
            'market': market
        }

    def _validate_inputs(self, sentiment: float, market_data: Dict) -> bool:
        """Validate signal inputs"""
        if not -1 <= sentiment <= 1:
            logger.warning(f"Invalid sentiment score: {sentiment}")
            return False
            
        required = ['volume', 'volatility', 'trend']
        missing = [k for k in required if k not in market_data]
        if missing:
            logger.warning(f"Missing market data: {missing}")
            return False
            
        return True

    def _neutral_signal(self) -> Dict:
        """Return neutral signal"""
        return {
            'direction': 0,
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': datetime.now()
        }

    def update_performance(self, signal: Dict, result: float) -> None:
        """Update signal performance tracking"""
        self.performance['total'] += 1
        
        # Check if signal was correct
        if signal['direction'] * result > 0:
            self.performance['correct'] += 1
            
        # Update accuracy
        self.performance['accuracy'] = (
            self.performance['correct'] / self.performance['total']
        )

    def get_performance(self) -> Dict:
        """Get signal performance metrics"""
        return {
            'total_signals': self.performance['total'],
            'accuracy': self.performance['accuracy'],
            'recent_signals': self.signals[-self.min_samples:]
        }
