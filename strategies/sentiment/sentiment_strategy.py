import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from models.llm.fingpt import FinGPT
from models.sentiment.analyzer import SentimentAnalyzer
from models.sentiment.preprocessor import TextPreprocessor
from strategies.base_strategy import BaseStrategy


class SentimentStrategy(BaseStrategy):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'strategies.yaml'
            
        # Load strategy config
        with open(config_path) as f:
            strategies_config = yaml.safe_load(f)
            self.config = strategies_config.get('sentiment', {})
            
        # Initialize strategy parameters
        self.processing = self.config.get('processing', {})
        self.metrics = self.config.get('metrics', {})
        self.signals = self.config.get('signals', {})
        self.trading = self.config.get('trading', {})
        self.risk = self.config.get('risk', {})
        
        # Validate configuration
        self._validate_config()
        logger.info("Sentiment strategy initialized with configuration")

        self.preprocessor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer(self.config.get("model_config", {}))
        self.sentiment_scores = {}
        self.signal_threshold = self.config.get("signal_threshold", 0.5)
        self.lookback_window = self.config.get("lookback_window", 24)
        self.market_data = {}
        self.active = True
        self.threshold = self.config.get("threshold", 0.3)
        self.lookback = self.config.get("lookback", 24)
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.fingpt = FinGPT(self.config.get("fingpt_config", {}))

    def _validate_config(self) -> None:
        """Validate strategy configuration"""
        required_sections = ['processing', 'metrics', 'signals', 'trading', 'risk']
        missing = [s for s in required_sections if not getattr(self, s)]
        if missing:
            raise ValueError(f"Missing required config sections: {', '.join(missing)}")
            
        # Validate signal parameters
        signals = self.signals
        assert 0 < signals.get('threshold', 0) <= 1.0, "Invalid signal threshold"
        assert signals.get('min_samples', 0) > 0, "Invalid min_samples"
        assert sum(signals.get('impact_weights', {}).values()) == 1.0, "Impact weights must sum to 1.0"

    async def process_market_data(self, data: Dict) -> None:
        """Process market data and news"""
        symbol = data.get("symbol")
        if not symbol:
            return

        price = data.get("price")
        volume = data.get("volume")

        # Update market data state
        self.market_data[symbol] = {
            "price": price,
            "volume": volume,
            "timestamp": data.get("timestamp"),
        }

        # Process sentiment
        news = data.get("news", [])
        if news:
            sentiment = await self._analyze_sentiment(news)
            self.sentiment_scores[symbol] = sentiment

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
                    "symbol": symbol,
                    "direction": "buy" if avg_score > 0 else "sell",
                    "strength": abs(avg_score),
                    "sentiment_score": avg_score,
                    "lookback_window": self.lookback_window,
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
        self.sentiment_scores[symbol] = self.sentiment_scores[symbol][
            -self.lookback_window :
        ]

    async def _fetch_latest_news(self, symbol: str) -> List[Dict]:
        """Fetch latest news articles for symbol"""
        try:
            news_service = self.config.get("news_service")
            if news_service:
                return await news_service.get_latest_news(
                    symbol=symbol, limit=self.config.get("news_limit", 10)
                )
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []

    async def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength"""
        base_size = self.config.get("base_position_size", 1000)
        max_size = self.config.get("max_position_size", 10000)

        # Scale by signal strength
        size = base_size * signal["strength"]

        # Apply limits
        return min(size, max_size)

    async def _validate_sentiment_data(self, scores: List[float]) -> bool:
        """Validate sentiment data quality"""
        if len(scores) < self.lookback_window / 2:
            return False
        if any(abs(score) > 1.0 for score in scores):
            return False
        return True

    async def on_trade(self, trade: Dict) -> None:
        """Handle trade updates"""
        symbol = trade.get("symbol")
        if symbol in self.positions:
            self.positions[symbol].update(trade)

    async def _analyze_sentiment(self, news: List[str]) -> float:
        """Analyze news sentiment using FinGPT"""
        scores = []
        for text in news:
            score = await self.fingpt.predict_sentiment(text)
            if score["confidence"] >= self.min_confidence:
                scores.append(score["sentiment"])
        return np.mean(scores) if scores else 0.0

    def generate_signal(self, sentiment_score: float, market_data: Dict) -> Dict:
        """Generate trading signal from sentiment and market data"""
        if not self._validate_signal_inputs(sentiment_score, market_data):
            return self._neutral_signal()
            
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            sentiment=sentiment_score,
            volume=market_data.get('volume', 0),
            correlation=market_data.get('correlation', 0)
        )
        
        # Apply confidence threshold
        if abs(signal_strength) < self.signals['threshold']:
            return self._neutral_signal()
            
        return {
            'direction': 1 if signal_strength > 0 else -1,
            'strength': abs(signal_strength),
            'confidence': min(abs(signal_strength) * 2, 1.0)
        }

    def _calculate_signal_strength(self, sentiment: float, 
                                 volume: float, correlation: float) -> float:
        """Calculate combined signal strength"""
        weights = self.signals['impact_weights']
        
        # Weighted combination of factors
        signal = (
            weights['sentiment'] * sentiment +
            weights['volume'] * (volume - 1.0) +  # Normalize around 1.0
            weights['correlation'] * correlation
        )
        
        return signal

    def _validate_signal_inputs(self, sentiment: float, market_data: Dict) -> bool:
        """Validate signal generation inputs"""
        if abs(sentiment) > 1.0:
            logger.warning(f"Invalid sentiment score: {sentiment}")
            return False
            
        if not market_data:
            logger.warning("Missing market data")
            return False
            
        return True

    def _neutral_signal(self) -> Dict:
        """Return neutral trading signal"""
        return {
            'direction': 0,
            'strength': 0.0,
            'confidence': 0.0
        }

    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate position size based on signal and risk parameters"""
        if signal['strength'] < self.signals['threshold']:
            return 0.0
            
        # Base position size
        position_size = self.trading['position_size'] * signal['confidence']
        
        # Apply risk limits
        position_size = min(
            position_size,
            portfolio_value * self.risk['max_concentration'],
            self.trading['max_position']
        )
        
        # Enforce minimum size
        if position_size < self.trading['position_size'] * 0.1:  # 10% of base size
            return 0.0
            
        return position_size
