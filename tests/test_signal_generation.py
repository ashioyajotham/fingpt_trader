import pytest
import numpy as np

import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.market.inefficiency import MarketInefficencyDetector
from models.sentiment.analyzer import SentimentAnalyzer

@pytest.fixture
def market_data():
    return {
        'prices': np.array([100, 101, 99, 102, 103]),
        'volume': np.array([1000, 1200, 800, 1500, 1300]),
        'sentiment': np.array([0.2, 0.3, -0.1, 0.4, 0.5])
    }

@pytest.fixture
def detector():
    return MarketInefficencyDetector(config={
        'lookback_periods': {'short': 5, 'medium': 20, 'long': 60},
        'min_confidence': 0.65
    })

def test_inefficiency_detection(detector, market_data):
    signals = detector.detect_inefficiencies(
        prices=market_data['prices'],
        volume=market_data['volume'],
        sentiment=market_data['sentiment']
    )
    assert isinstance(signals, dict)
    assert 'confidence' in signals
    assert 0 <= signals['confidence'] <= 1

@pytest.mark.asyncio
async def test_sentiment_analysis():
    analyzer = SentimentAnalyzer({
        'model_name': 'FinGPT/fingpt-mt_falcon-7b',
        'batch_size': 16
    })
    
    test_headlines = [
        "Company XYZ reports record profits",
        "Market faces uncertainty amid economic data"
    ]
    
    sentiment = await analyzer.analyze_text(test_headlines)
    assert isinstance(sentiment, np.ndarray)
    assert len(sentiment) == len(test_headlines)
    assert all(-1 <= s <= 1 for s in sentiment)
