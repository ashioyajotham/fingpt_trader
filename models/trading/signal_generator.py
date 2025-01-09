from typing import Dict
import numpy as np
from .market_inefficiency.detector import MarketInefficiencyDetector

class SignalGenerator:
    def __init__(self, sentiment_threshold: float = 0.5):
        self.sentiment_threshold = sentiment_threshold
        self.inefficiency_detector = MarketInefficiencyDetector()
        
    def generate_signal(self, 
                       sentiment_score: float,
                       market_data: Dict) -> Dict[str, float]:
        """Generate trading signals based on sentiment and market inefficiencies"""
        # Detect market inefficiencies
        micro_patterns = self.inefficiency_detector.detect_microstructure_patterns(
            market_data['prices'], 
            market_data['volumes']
        )
        psych_patterns = self.inefficiency_detector.detect_psychology_patterns(
            market_data['prices']
        )
        
        # Combine signals
        signal = {
            'direction': 0,
            'confidence': 0.0,
            'strength': 0.0
        }
        
        # Weight different factors
        sentiment_weight = 0.3
        micro_weight = 0.4
        psych_weight = 0.3
        
        # Calculate composite signal
        composite_score = (
            sentiment_score * sentiment_weight +
            np.mean(list(micro_patterns.values())) * micro_weight +
            np.mean(list(psych_patterns.values())) * psych_weight
        )
        
        if abs(composite_score) > self.sentiment_threshold:
            signal['direction'] = np.sign(composite_score)
            signal['confidence'] = abs(composite_score)
            signal['strength'] = min(1.0, abs(composite_score) * 1.5)
            
        return signal