import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import numpy as np
import pandas as pd

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from llm.utils.tokenizer import TokenizerConfig

from models.llm.fingpt import FinGPT

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_config: Dict):
        self.config = model_config or {} # gets from config/ directory
        self.fingpt = FinGPT(self.config.get("fingpt_config", {}))
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.batch_size = model_config.get("batch_size", 16)

    async def initialize(self):
        """Initialize analyzer resources"""
        try:
            await self.fingpt.initialize()
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Analyzer initialization failed: {str(e)}")
            raise

    async def analyze(self, texts: List[str]) -> Dict:
        scores = []
        confidences = []

        for text in texts:
            result = await self.fingpt.predict_sentiment(text)
            if result["confidence"] >= self.min_confidence:
                scores.append(result["sentiment"])
                confidences.append(result["confidence"])

        return {
            "sentiment": np.mean(scores) if scores else 0.0,
            "confidence": np.mean(confidences) if confidences else 0.0,
            "samples": len(scores),
        }

    def analyze_text(self, text: Union[str, List[str]]) -> Dict:
        """Analyze sentiment of single text or list of texts"""
        if isinstance(text, str):
            text = [text]

        sentiments = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i : i + self.batch_size]
            batch_sentiments = self.model.predict_sentiment(batch)
            sentiments.extend(batch_sentiments)

        return {
            "sentiments": sentiments,
            "timestamp": datetime.now().isoformat(),
            "summary": self._summarize_sentiments(sentiments),
        }

    def analyze_news_feed(self, news_items: List[Dict]) -> pd.DataFrame:
        """Analyze sentiment for news feed"""
        texts = [
            item["title"] + " " + item.get("description", "") for item in news_items
        ]

        results = self.analyze_text(texts)

        df = pd.DataFrame(
            {
                "timestamp": [item.get("timestamp") for item in news_items],
                "text": texts,
                "sentiment": results["sentiments"],
                "source": [item.get("source") for item in news_items],
            }
        )

        return df

    def _summarize_sentiments(self, sentiments: List[str]) -> Dict:
        """Generate sentiment summary statistics"""
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        scores = [sentiment_map[s] for s in sentiments]

        return {
            "mean_score": np.mean(scores),
            "sentiment_counts": {
                label: sentiments.count(label) for label in sentiment_map.keys()
            },
            "majority_sentiment": max(set(sentiments), key=sentiments.count),
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'fingpt'):
                await self.fingpt.cleanup()
        except Exception as e:
            logger.error(f"Sentiment analyzer cleanup failed: {str(e)}")
