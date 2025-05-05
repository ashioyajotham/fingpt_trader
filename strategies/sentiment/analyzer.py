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
from utils.logging import debug, info, warning, error

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, config):
        # Initialize FinGPT model from config
        fingpt_config = config.get('model_config', {})
        self.fingpt = FinGPT(fingpt_config)  # Create the model instance
        
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
        self.sentiment_window = config.get('sentiment_window', 24)  # Default to 24 hours
        
        # Add missing threshold parameters
        self.detection_threshold = config.get('detection_threshold', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Other existing attributes
        self.min_correlation_samples = config.get('min_correlation_samples', 10)
        self.price_impact_threshold = config.get('price_impact_threshold', 0.01)
        self.active_pairs = market_feed_config['pairs']
        
        # Data handlers
        self.handlers = {
            'market': self._handle_market_data,
            'news': self._handle_news_data
        }
        
        # State management
        self.last_update = datetime.now()
        self.sentiment_history = {}
        self.market_correlation = {}
        self.last_market_update = datetime.now() 
        self.last_news_update = datetime.now()
        
        info("Initializing Sentiment Analyzer")
        debug("Loading sentiment model", {
            "model_name": config.get("model_name"),
            "threshold": config.get("detection_threshold")
        })

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

    async def analyze(self, text: str) -> Dict:
        """Analyze sentiment using LLM"""
        try:
            # Check if FinGPT is properly initialized
            if not hasattr(self, 'fingpt') or self.fingpt is None:
                logger.error("FinGPT model not initialized")
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            # Improve the prompt for better sentiment differentiation
            prompt = f"""
            Analyze the sentiment of the following financial news text. 
            Consider market impact, investor sentiment, and financial implications.
            Rate on a scale from -1.0 (extremely bearish) to 1.0 (extremely bullish).
            Provide only a JSON response with 'sentiment' and 'confidence' values.
            
            News text: {text}
            
            JSON response:
            """
            
            logger.debug(f"Sending prompt to FinGPT: {prompt[:100]}...")
            response = await self.fingpt.generate(prompt, temperature=0.2)
            logger.debug(f"Received raw response: {response[:150]}...")
            
            # Parse the response
            try:
                # Extract JSON from response
                import json
                import re
                
                # Find JSON content in response
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    # Use sentiment key if it exists, otherwise default to 0.0
                    sentiment_score = result.get('sentiment', 0.0)
                    confidence = result.get('confidence', 0.0)
                    
                    # Log detailed results
                    try:
                        sentiment_score = float(sentiment_score)
                        confidence = float(confidence)
                        logger.info(f"Sentiment analysis: score={sentiment_score:.2f}, confidence={confidence:.2f}")
                    except (ValueError, TypeError):
                        logger.info(f"Sentiment analysis: score={sentiment_score} (invalid format), confidence={confidence}")
                    
                    if abs(sentiment_score) >= self.detection_threshold and confidence >= self.confidence_threshold:
                        logger.info(f"Strong sentiment signal detected! (threshold={self.detection_threshold:.2f})")
                    else:
                        logger.info(f"Sentiment below thresholds, no signal generated")
                    
                    return result
            except Exception as json_error:
                logger.error(f"Error parsing sentiment JSON: {json_error}")
            
            # Fallback return if parsing fails
            return {'sentiment': 0.0, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'sentiment': 0.0, 'confidence': 0.0}

    async def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment score and confidence
        """
        try:
            # Use the existing analyze method
            result = await self.analyze(text)
            return result
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}")
            # Return neutral sentiment with low confidence on error
            return {
                'sentiment': 0.0,
                'confidence': 0.1
            }

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

    async def process_market_data(self, data: Dict) -> None:
        """Process market data for sentiment analysis"""
        try:
            # Process each symbol in the data
            for symbol, market_data in data.items():
                # Skip if no price data
                if not market_data or not market_data.get('price'):
                    continue
                
                price = float(market_data.get('price', 0))
                timestamp = datetime.now()
                
                # Add to price history - FIXED: remove await
                self.add_market_data(symbol, price, timestamp)
                
                # Check for relevant news and analyze
                news_items = await self._fetch_relevant_news(symbol)
                for news in news_items:
                    await self.analyze_sentiment(news, symbol)  # Now this will work
                    
                # Update correlation metrics
                await self.update_correlation(symbol)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")

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

    async def update_correlation(self, symbol):
        """Calculate correlation between sentiment and price movements"""
        try:
            # Get historical sentiment data for this symbol
            if symbol not in self.sentiment_history:
                logger.debug(f"No sentiment history for {symbol}, skipping correlation")
                return
                
            # Extract sentiment changes from history
            sentiment_data = self.sentiment_history[symbol]
            if len(sentiment_data) < self.min_correlation_samples:
                logger.debug(f"Not enough samples for {symbol} correlation: {len(sentiment_data)}/{self.min_correlation_samples}")
                return
                
            # Calculate sentiment changes
            sentiment_values = [entry['score'] for entry in sentiment_data]
            sentiment_changes = [sentiment_values[i] - sentiment_values[i-1] 
                                for i in range(1, len(sentiment_values))]
                                
            # Get price data for correlation
            if not hasattr(self, 'price_history') or symbol not in self.price_history:
                logger.debug(f"No price history for {symbol}, skipping correlation")
                return
                
            # Calculate price changes
            price_data = self.price_history[symbol]
            price_values = [entry['price'] for entry in price_data]
            price_changes = [price_values[i]/price_values[i-1] - 1 
                            for i in range(1, len(price_values))]
                            
            # Ensure we have matching data points
            min_length = min(len(sentiment_changes), len(price_changes))
            if min_length < self.min_correlation_samples:
                logger.debug(f"Insufficient matching data points for correlation: {min_length}")
                return
                
            # Calculate correlation
            import numpy as np
            correlation = np.corrcoef(
                sentiment_changes[:min_length],
                price_changes[:min_length]
            )[0, 1]
            
            # Update correlation record
            self.market_correlation[symbol] = {
                'value': correlation,
                'updated_at': datetime.now(),
                'samples': min_length
            }
            
            logger.info(f"Sentiment-price correlation for {symbol}: {correlation:.4f} (samples: {min_length})")
            
        except Exception as e:
            logger.error(f"Error updating correlation for {symbol}: {e}")

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
        """Process incoming news data for sentiment analysis"""
        try:
            # Extract text from news item
            text = f"{news_item.get('title', '')}. {news_item.get('content', '')}"
            
            # With await
            result = await self.analyze(text)
            
            # Check if result is None before accessing
            if result is None:
                logger.warning("Sentiment analysis returned None")
                return
                
            score = result.get('sentiment', 0.0)  # Changed from 'compound'
            confidence = result.get('confidence', 0.0)
            
            # Store in sentiment history
            for symbol in news_item.get('symbols', []):
                if symbol not in self.sentiment_scores:
                    self.sentiment_scores[symbol] = []
                    
                self.sentiment_scores[symbol].append({
                    'score': score,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"Error processing news for sentiment: {e}")

    async def _fetch_relevant_news(self, pair: str) -> List[str]:
        """Fetch relevant news for the trading pair"""
        try:
            base_asset = pair.replace('USDT', '').replace('USD', '')
            
            # Use news_feed instead of news_service
            news_data = await self.news_feed.get_latest()
            
            # Filter news items relevant to this asset
            relevant_news = []
            for item in news_data:
                content = f"{item.get('title', '')} {item.get('content', '')}"
                if base_asset.lower() in content.lower():
                    relevant_news.append(content)
            
            logger.info(f"Fetched {len(relevant_news)} relevant news items for {pair}")
            return relevant_news
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {pair}: {e}")
            return []

    async def analyze_sentiment(self, text: str, symbol: str) -> None:
        """Analyze sentiment for a specific symbol and store results"""
        try:
            # Use existing analyze method
            result = await self.analyze(text)
            
            # Store sentiment data
            if symbol not in self.sentiment_scores:
                self.sentiment_scores[symbol] = []
                
            self.sentiment_scores[symbol].append({
                'score': result.get('sentiment', 0.0),  # Use 'sentiment' key instead of 'compound'
                'confidence': result.get('confidence', 0.0),
                'timestamp': datetime.now()
            })
            
            # Log sentiment for debugging
            logger.debug(f"Sentiment for {symbol}: {result.get('sentiment', 0.0):.2f} (confidence: {result.get('confidence', 0.0):.2f})")
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
