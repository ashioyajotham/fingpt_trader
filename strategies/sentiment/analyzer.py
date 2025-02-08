from typing import Dict, List, Optional, Tuple

import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..base_strategy import BaseStrategy


class SentimentAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.vader = SentimentIntensityAnalyzer()

        # Financial term modifiers
        self.term_scores = {
            "buy": 0.5,
            "sell": -0.5,
            "bullish": 0.7,
            "bearish": -0.7,
            "long": 0.3,
            "short": -0.3,
            "upgrade": 0.4,
            "downgrade": -0.4,
        }

        # Add financial terms to VADER lexicon
        self.vader.lexicon.update(self.term_scores)

    async def initialize(self) -> None:
        """Initialize the analyzer - required by system interface"""
        pass  # Nothing to initialize for VADER/TextBlob

    async def cleanup(self) -> None:
        """Cleanup resources - required by system interface"""
        pass  # Nothing to cleanup for VADER/TextBlob

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment using multiple models"""
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)

        # Get TextBlob sentiment
        blob = TextBlob(text)
        blob_scores = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

        # Combine scores with weights
        combined_score = self._combine_scores(vader_scores, blob_scores)

        return {
            "compound": combined_score,
            "vader": vader_scores,
            "textblob": blob_scores,
            "confidence": self._calculate_confidence(vader_scores, blob_scores),
        }

    def _combine_scores(self, vader: Dict, blob: Dict) -> float:
        """Combine scores from different models"""
        # Weights for each model
        vader_weight = 0.7
        blob_weight = 0.3

        combined = vader_weight * vader["compound"] + blob_weight * blob["polarity"]

        return np.clip(combined, -1.0, 1.0)

    def _calculate_confidence(self, vader: Dict, blob: Dict) -> float:
        """Calculate confidence score for sentiment"""
        # Check agreement between models
        score_diff = abs(vader["compound"] - blob["polarity"])
        agreement_factor = 1.0 - (score_diff / 2.0)

        # Consider subjectivity
        subjectivity_factor = 1.0 - blob["subjectivity"]

        confidence = (agreement_factor + subjectivity_factor) / 2.0
        return confidence


class SentimentStrategy(BaseStrategy):
    """
    Sentiment-based trading strategy.
    Combines news sentiment with market data analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None, profile: Optional[Dict] = None):
        super().__init__(config, profile)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.5)
        self.sentiment_window = config.get('sentiment_window', 24)  # hours
        
    async def _generate_base_signals(self, market_data: Dict) -> List[Dict]:
        signals = []
        
        for pair in self.active_pairs:
            if pair not in self.market_state['sentiment']:
                continue
                
            sentiment_score = self._calculate_sentiment_score(pair)
            market_impact = self._estimate_market_impact(pair, market_data)
            
            if abs(sentiment_score) > self.sentiment_threshold:
                signals.append({
                    'symbol': pair,
                    'direction': np.sign(sentiment_score),
                    'strength': min(abs(sentiment_score) * market_impact, 1.0),
                    'sentiment': sentiment_score,
                    'market_impact': market_impact
                })
                
        return signals
        
    def _calculate_sentiment_score(self, pair: str) -> float:
        """Calculate weighted sentiment score"""
        sentiments = self.market_state['sentiment'].get(pair, [])
        if not sentiments:
            return 0.0
            
        # Weight recent sentiment more heavily
        weights = np.exp(-np.arange(len(sentiments)) / self.sentiment_window)
        weighted_score = np.average([s.get('score', 0) for s in sentiments], weights=weights)
        
        return weighted_score
        
    def _estimate_market_impact(self, pair: str, market_data: Dict) -> float:
        """Estimate potential market impact of sentiment"""
        if pair not in market_data.get('volume', {}):
            return 0.5  # Default impact if no volume data
            
        # Calculate relative volume
        volumes = market_data['volume'][pair]
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-24:])  # 24-hour average
        
        # Normalize impact between 0 and 1
        impact = min(current_vol / (avg_vol + 1e-10), 2.0) / 2.0
        
        return impact
