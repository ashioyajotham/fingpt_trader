from typing import Dict, List, Optional
import pandas as pd
from ...strategies.base_strategy import BaseStrategy
from ...models.sentiment.preprocessor import TextPreprocessor
from ...models.sentiment.analyzer import SentimentAnalyzer

class SentimentStrategy(BaseStrategy):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
        self.sentiment_scores = {}
        self.signal_threshold = self.config.get('signal_threshold', 0.5)
        self.lookback_window = self.config.get('lookback_window', 24)
        self.market_data = {}
        self.active = True
        
    async def process_market_data(self, data: Dict) -> None:
        """Process market data update"""
        symbol = data.get('symbol')
        if not symbol:
            return
            
        price = data.get('price')
        volume = data.get('volume')
        
        # Update market data state
        self.market_data[symbol] = {
            'price': price,
            'volume': volume,
            'timestamp': data.get('timestamp')
        }
        
        # Process sentiment
        await self._update_sentiment(symbol)
        
        # Generate new signals
        if self.active:
            await self.generate_signals()
        
    async def generate_signals(self) -> List[Dict]:
        """Generate trading signals based on sentiment"""
        signals = []
        for symbol, scores in self.sentiment_scores.items():
            if len(scores) < self.lookback_window:
                continue
                
            avg_score = sum(scores) / len(scores)
            if abs(avg_score) > self.signal_threshold:
                signal = {
                    'symbol': symbol,
                    'direction': 'buy' if avg_score > 0 else 'sell',
                    'strength': abs(avg_score),
                    'sentiment_score': avg_score,
                    'lookback_window': self.lookback_window
                }
                signals.append(signal)
                
        self.signals = signals
        return signals
        
    async def _update_sentiment(self, symbol: str) -> None:
        """Update sentiment scores for symbol"""
        news = await self._fetch_latest_news(symbol)
        if not news:
            return
            
        processed_text = self.preprocessor.preprocess(news)
        score = self.analyzer.analyze(processed_text)
        
        if symbol not in self.sentiment_scores:
            self.sentiment_scores[symbol] = []
            
        self.sentiment_scores[symbol].append(score)
        
        # Keep only lookback window worth of scores
        self.sentiment_scores[symbol] = \
            self.sentiment_scores[symbol][-self.lookback_window:]
            
    async def _fetch_latest_news(self, symbol: str) -> List[Dict]:
        """Fetch latest news articles for symbol"""
        try:
            news_service = self.config.get('news_service')
            if news_service:
                return await news_service.get_latest_news(
                    symbol=symbol,
                    limit=self.config.get('news_limit', 10)
                )
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []

    async def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength"""
        base_size = self.config.get('base_position_size', 1000)
        max_size = self.config.get('max_position_size', 10000)
        
        # Scale by signal strength
        size = base_size * signal['strength']
        
        # Apply limits
        return min(size, max_size)

    async def _validate_sentiment_data(self, scores: List[float]) -> bool:
        """Validate sentiment data quality"""
        if len(scores) < self.lookback_window / 2:
            return False
        if any(abs(score) > 1.0 for score in scores):
            return False
        return True