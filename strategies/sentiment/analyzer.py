from typing import Dict, List, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from models.llm.fingpt import FinGPT
from services.data_feeds.news_service import NewsDataFeed
from services.data_feeds.market_data_service import MarketDataFeed
from services.base_service import BaseService

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        
        # Get model instance from config
        self.fingpt = config.get('model')
        if not self.fingpt:
            raise ValueError("FinGPT model instance required")
            
        # Initialize data feeds
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
        
        # State management
        self.last_update = datetime.now()
        self.sentiment_history = {}
        self.market_correlation = {}

    async def _setup(self) -> None:
        """Required implementation of abstract method"""
        try:
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
        except Exception as e:
            logger.error(f"Sentiment analyzer setup failed: {e}")
            raise

    async def _cleanup(self) -> None:
        """Required implementation of abstract method"""
        try:
            await self.market_feed.stop()
            await self.news_feed.stop()
            
            # Clear state
            self.sentiment_buffer.clear()
            self.price_history.clear()
            self.sentiment_scores.clear()
            
            logger.info("Sentiment analyzer cleaned up")
        except Exception as e:
            logger.error(f"Sentiment analyzer cleanup failed: {e}")
            raise

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
        """Analyze sentiment using FinGPT with timeout handling"""
        if not text or len(text.strip()) < 10:
            logger.warning("Empty or too short text for sentiment analysis")
            return {'compound': 0.0, 'confidence': 0.0}
            
        chunks = self._chunk_text(text, max_tokens=750)
        logger.info(f"Processing {len(chunks)} text chunks for sentiment analysis")
        
        sentiments = []
        async with asyncio.timeout(30):  # Add 30 second timeout
            try:
                for i, chunk in enumerate(chunks):
                    try:
                        sentiment = await self.fingpt.predict_sentiment(chunk)
                        sentiments.append(sentiment)
                        logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                        continue
                    
            except asyncio.TimeoutError:
                logger.error("Sentiment analysis timed out after 30 seconds")
                return {'compound': 0.0, 'confidence': 0.0}
                
            except Exception as e:
                logger.error(f"Sentiment analysis error: {str(e)}")
                return {'compound': 0.0, 'confidence': 0.0}
        
        if not sentiments:
            logger.warning("No valid sentiment chunks processed, returning neutral")
            return {'compound': 0.0, 'confidence': 0.0}
            
        # Average sentiments with error handling
        try:
            compound = sum(s['sentiment'] for s in sentiments) / len(sentiments)
            confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            logger.info(f"Sentiment analysis complete: score={compound:.2f}, confidence={confidence:.2f}")
            return {
                'compound': compound,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating final sentiment: {str(e)}")
            return {'compound': 0.0, 'confidence': 0.0}

    def _chunk_text(self, text: str, max_tokens: int = 750) -> List[str]:
        """Split text into chunks for processing"""
        # First clean the text to remove model output patterns
        cleaned_text = ""
        for line in text.split('\n'):
            if not any(pattern in line.lower() for pattern in [
                'sentiment score:', 'you:', 'assistant:', 
                'analysis:', 'raw response'
            ]):
                cleaned_text += line + " "
        
        # Now continue with the chunking as before
        words = cleaned_text.split()
        chunks = []
        words_per_chunk = max_tokens // 6  # Conservative estimate
        
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            if len(chunk) > 10:
                chunks.append(chunk)
        
        return chunks

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

    async def _fetch_relevant_news(self, pair: str) -> List[str]:
        """Fetch relevant news for the trading pair"""
        try:
            base_asset = pair.replace('USDT', '').replace('USD', '')
            
            # Broader search terms
            search_terms = {
                'BTC': ['Bitcoin', 'BTC', 'crypto market', 'cryptocurrency'],
                'ETH': ['Ethereum', 'ETH', 'DeFi', 'smart contracts'],
                'BNB': ['Binance', 'BNB', 'exchange token']
            }
            
            # Use multiple search terms for better coverage
            news_data = []
            terms = search_terms.get(base_asset, [f"{base_asset} crypto"])
            
            for term in terms:
                articles = await self.news_service.get_news(term)
                news_data.extend(articles)
            
            # Deduplicate articles
            seen = set()
            unique_texts = []
            
            for article in news_data:
                title = article.get('title', '')
                if title and title not in seen:
                    seen.add(title)
                    unique_texts.append(title)
                    if article.get('description'):
                        unique_texts.append(article['description'])
            
            logger.info(f"Fetched {len(unique_texts)} unique news items for {pair}")
            return unique_texts
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {pair}: {e}")
            return []
