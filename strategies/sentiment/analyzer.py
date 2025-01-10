from typing import Dict, List, Optional, Tuple
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.vader = SentimentIntensityAnalyzer()
        
        # Financial term modifiers
        self.term_scores = {
            'buy': 0.5,
            'sell': -0.5,
            'bullish': 0.7,
            'bearish': -0.7,
            'long': 0.3,
            'short': -0.3,
            'upgrade': 0.4,
            'downgrade': -0.4
        }
        
        # Add financial terms to VADER lexicon
        self.vader.lexicon.update(self.term_scores)
        
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment using multiple models"""
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        blob_scores = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Combine scores with weights
        combined_score = self._combine_scores(vader_scores, blob_scores)
        
        return {
            'compound': combined_score,
            'vader': vader_scores,
            'textblob': blob_scores,
            'confidence': self._calculate_confidence(vader_scores, blob_scores)
        }
        
    def _combine_scores(self, vader: Dict, blob: Dict) -> float:
        """Combine scores from different models"""
        # Weights for each model
        vader_weight = 0.7
        blob_weight = 0.3
        
        combined = (
            vader_weight * vader['compound'] + 
            blob_weight * blob['polarity']
        )
        
        return np.clip(combined, -1.0, 1.0)
        
    def _calculate_confidence(self, vader: Dict, blob: Dict) -> float:
        """Calculate confidence score for sentiment"""
        # Check agreement between models
        score_diff = abs(vader['compound'] - blob['polarity'])
        agreement_factor = 1.0 - (score_diff / 2.0)
        
        # Consider subjectivity
        subjectivity_factor = 1.0 - blob['subjectivity']
        
        confidence = (agreement_factor + subjectivity_factor) / 2.0
        return confidence