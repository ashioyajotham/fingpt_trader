from typing import Dict, List, Optional, Tuple
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pandas as pd

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

        # Add real-time sentiment tracking
        self.sentiment_window = config.get('sentiment_window', 24)  # hours
        self.sentiment_history = {}
        self.market_correlation = {}
        self.min_samples = config.get('min_samples', 10)
        
        # Real-time correlation tracking
        self.correlation_window = config.get('correlation_window', 12)  # hours
        self.min_correlation_samples = config.get('min_correlation_samples', 24)
        self.price_impact_threshold = config.get('price_impact_threshold', 0.02)
        
        # Initialize state
        self.price_impacts = {}
        self.correlation_history = {}

    async def initialize(self) -> None:
        """Initialize the analyzer - required by system interface"""
        self.last_update = datetime.now()
        self.sentiment_buffer = {}
        
        # Initialize market correlation tracking
        self.price_history = {}
        self.sentiment_scores = {}

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

    def add_market_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Add market data for sentiment correlation"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=self.sentiment_window)
        self.price_history[symbol] = [
            p for p in self.price_history[symbol] 
            if p['timestamp'] > cutoff
        ]

    def add_sentiment_data(self, symbol: str, text: str, timestamp: datetime) -> None:
        """Process and store new sentiment data"""
        sentiment = self.analyze(text)
        
        if symbol not in self.sentiment_scores:
            self.sentiment_scores[symbol] = []
            
        self.sentiment_scores[symbol].append({
            'score': sentiment['compound'],
            'timestamp': timestamp
        })
        
        # Keep only recent scores
        cutoff = datetime.now() - timedelta(hours=self.sentiment_window)
        self.sentiment_scores[symbol] = [
            s for s in self.sentiment_scores[symbol]
            if s['timestamp'] > cutoff
        ]
        
        # Update correlation if enough data
        self._update_correlation(symbol)

    def _update_correlation(self, symbol: str) -> None:
        """Calculate sentiment-price correlation"""
        if (symbol not in self.price_history or 
            symbol not in self.sentiment_scores or
            len(self.sentiment_scores[symbol]) < self.min_samples):
            return
            
        # Create time-aligned series
        df = pd.DataFrame(self.sentiment_scores[symbol])
        df.set_index('timestamp', inplace=True)
        
        price_df = pd.DataFrame(self.price_history[symbol])
        price_df.set_index('timestamp', inplace=True)
        
        # Resample to common timeframe
        aligned = pd.merge_asof(
            df, price_df,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta('5min')
        )
        
        if len(aligned) >= self.min_samples:
            self.market_correlation[symbol] = aligned['score'].corr(
                aligned['price'].pct_change()
            )

    async def calculate_price_impact(self, sentiment_score: float, 
                                   symbol: str, price_data: List[Dict]) -> float:
        """Calculate estimated price impact of sentiment"""
        if not price_data or symbol not in self.correlation_history:
            return 0.0
            
        # Get historical correlation
        correlation = self.correlation_history[symbol].get('value', 0.5)
        
        # Calculate expected impact
        impact = sentiment_score * correlation * self.price_impact_threshold
        
        # Store impact
        self.price_impacts[symbol] = {
            'score': sentiment_score,
            'impact': impact,
            'timestamp': datetime.now()
        }
        
        return impact

    async def update_correlation(self, symbol: str, sentiment_changes: pd.Series, 
                               price_changes: pd.Series) -> None:
        """Update price-sentiment correlation"""
        if len(sentiment_changes) < self.min_correlation_samples:
            return
            
        # Calculate rolling correlation
        correlation = sentiment_changes.rolling(
            window=self.min_correlation_samples
        ).corr(price_changes)
        
        # Store correlation
        self.correlation_history[symbol] = {
            'value': correlation.iloc[-1],
            'timestamp': datetime.now(),
            'samples': len(sentiment_changes)
        }

    def get_sentiment_signal(self, symbol: str) -> Dict:
        """Get trading signal from sentiment"""
        impact = self.price_impacts.get(symbol, {}).get('impact', 0)
        score = self.price_impacts.get(symbol, {}).get('score', 0)
        
        return {
            'direction': np.sign(score),
            'strength': abs(impact),
            'confidence': self._calculate_confidence(
                self.vader.polarity_scores(str(score)),
                {'polarity': score, 'subjectivity': 0.5}
            )
        }

    def get_sentiment_impact(self, symbol: str) -> float:
        """Get sentiment impact score"""
        if symbol not in self.sentiment_scores:
            return 0.0
            
        recent_scores = [
            s['score'] for s in self.sentiment_scores[symbol]
            if s['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_scores:
            return 0.0
            
        # Weight by correlation if available
        base_sentiment = np.mean(recent_scores)
        correlation = self.market_correlation.get(symbol, 0.5)
        
        return base_sentiment * correlation
