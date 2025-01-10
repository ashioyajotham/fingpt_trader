from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

class SignalGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.signals_history = []
        self.threshold = self.config.get('threshold', 0.3)
        self.time_decay = self.config.get('time_decay', 0.95)
        self.min_samples = self.config.get('min_samples', 5)
        
    def generate_signal(self, 
                       sentiment_data: List[Dict],
                       market_data: Optional[Dict] = None) -> Optional[Dict]:
        """Generate trading signal from sentiment data"""
        if len(sentiment_data) < self.min_samples:
            return None
            
        # Calculate time-weighted sentiment score
        weighted_score = self._calculate_weighted_score(sentiment_data)
        
        # Calculate signal strength and confidence
        strength = self._calculate_strength(weighted_score)
        confidence = self._calculate_confidence(sentiment_data)
        
        if abs(weighted_score) < self.threshold:
            return None
            
        signal = {
            'timestamp': datetime.now(),
            'direction': 'long' if weighted_score > 0 else 'short',
            'strength': strength,
            'confidence': confidence,
            'sentiment_score': weighted_score,
            'sample_size': len(sentiment_data)
        }
        
        self.signals_history.append(signal)
        return signal
        
    def _calculate_weighted_score(self, data: List[Dict]) -> float:
        """Calculate time-weighted sentiment score"""
        scores = []
        weights = []
        
        for i, item in enumerate(data):
            scores.append(item['compound'])
            weights.append(self.time_decay ** i)
            
        return np.average(scores, weights=weights)
        
    def _calculate_strength(self, score: float) -> float:
        """Calculate signal strength (0-1)"""
        return min(abs(score) / self.threshold, 1.0)
        
    def _calculate_confidence(self, data: List[Dict]) -> float:
        """Calculate signal confidence"""
        confidences = [d.get('confidence', 0.5) for d in data]
        return np.mean(confidences)