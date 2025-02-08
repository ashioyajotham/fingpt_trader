from typing import Dict, List, Optional, Tuple
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pandas as pd
from models.llm.fingpt import FinGPT
from services.data_feeds.news_service import NewsDataFeed
from services.data_feeds.market_data_service import MarketDataFeed

import logging
logger = logging.getLogger(__name__)

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

        # Get model instance from config
        self.fingpt = config.get('model')
        if not self.fingpt:
            raise ValueError("FinGPT model instance required")
            
        # Get model config
        self.model_config = config.get('model_config', {})
        
        # Initialize data feeds with proper config
        market_feed_config = {
            'pairs': config.get('pairs', ['BTCUSDT', 'ETHUSDT']),
            'update_interval': config.get('market_interval', 60),
            'cache_size': config.get('cache_size', 1000)
        }
        
        news_feed_config = {
            'update_interval': config.get('news_interval', 300),
            'sources': config.get('news_sources', []),
            'languages': ['en'],
            'relevance_threshold': 0.5
        }
        
        self.market_feed = MarketDataFeed(market_feed_config)
        self.news_feed = NewsDataFeed(news_feed_config)
        
        # Data handlers
        self.handlers = {
            'market': self._handle_market_data,
            'news': self._handle_news_data
        }
        
        # Sentiment aggregation
        self.sentiment_scores = {
            'vader': 0.3,    # Traditional NLP weight
            'textblob': 0.2, # Basic sentiment weight
            'fingpt': 0.5    # FinGPT model weight
        }
        
        # Configure data update frequencies
        self.news_interval = config.get('news_interval', 300)  # 5 minutes
        self.market_interval = config.get('market_interval', 60)  # 1 minute
        
        # Last update timestamps
        self.last_news_update = datetime.now()
        self.last_market_update = datetime.now()

    async def initialize(self) -> None:
        """Initialize analyzer and data feeds"""
        self.last_update = datetime.now()
        self.sentiment_buffer = {}
        
        # Initialize market correlation tracking
        self.price_history = {}
        self.sentiment_scores = {}
        
        # Initialize data feeds
        await self.market_feed.start()
        await self.market_feed.subscribe(self.handlers['market'])
        
        await self.news_feed.start()
        await self.news_feed.subscribe(self.handlers['news'])
        
        logger.info("Sentiment analyzer initialized with FinGPT and data feeds")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            await self.market_feed.stop()
            await self.news_feed.stop()
            logger.info("Data feeds stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def analyze(self, text: str) -> Dict[str, float]:
        """Enhanced sentiment analysis with FinGPT"""
        # Use existing FinGPT instance
        sentiment = await self.fingpt.predict_sentiment(text)
        
        # Get traditional sentiment as backup
        basic_sentiment = self._get_basic_sentiment(text)
        
        # Combine with proper weights
        combined = (
            self.sentiment_scores['fingpt'] * sentiment +
            self.sentiment_scores['vader'] * basic_sentiment['vader']['compound'] +
            self.sentiment_scores['textblob'] * basic_sentiment['textblob']['polarity']
        )
        
        return {
            'compound': combined,
            'components': {
                'fingpt': sentiment,
                'basic': basic_sentiment
            }
        }

    async def update_market_data(self) -> None:
        """Update market data from feed"""
        if (datetime.now() - self.last_market_update).seconds > self.market_interval:
            market_data = await self.market_feed.get_latest()
            for symbol, data in market_data.items():
                self.add_market_data(symbol, data['price'], data['timestamp'])
            self.last_market_update = datetime.now()

    async def update_news_data(self) -> None:
        """Update news data from feed"""
        if (datetime.now() - self.last_news_update).seconds > self.news_interval:
            news_data = await self.news_feed.get_latest()
            for item in news_data:
                await self.process_news_item(item)
            self.last_news_update = datetime.now()

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

    async def _handle_market_data(self, event_type: str, pair: str, data: Dict) -> None:
        """Process market data updates"""
        if event_type == 'orderbook':
            # Update market state for sentiment correlation
            price = float(data['data']['asks'][0][0])  # Best ask price
            await self.add_market_data(pair, price, datetime.now())
            
        elif event_type == 'trades':
            # Process trade data for market impact analysis
            await self._process_trade_impact(pair, data['data'])

    async def _handle_news_data(self, news_item: Dict) -> None:
        """Process news updates"""
        # Extract relevant symbols
        symbols = news_item.get('symbols', [])
        
        for symbol in symbols:
            if symbol in self.active_pairs:
                # Process sentiment for each relevant pair
                await self.add_sentiment_data(
                    symbol,
                    f"{news_item['title']} {news_item['content']}",
                    news_item['timestamp']
                )
