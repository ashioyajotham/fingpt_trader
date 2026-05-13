import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import os

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
        self.fingpt = None
        if os.getenv("HUGGINGFACE_TOKEN") or self.config.get("fingpt_config", {}).get("offline"):
            self.fingpt = FinGPT(self.config.get("fingpt_config", {}))
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.batch_size = model_config.get("batch_size", 16)

    async def initialize(self):
        """Initialize analyzer resources"""
        try:
            if self.fingpt:
                await self.fingpt.initialize()
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Analyzer initialization failed: {str(e)}")
            raise

    async def analyze(self, texts: List[str]) -> Dict:
        scores = []
        confidences = []

        for text in texts:
            if self.fingpt:
                result = await self.fingpt.predict_sentiment(text)
            else:
                result = self._lexical_sentiment(text)
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
        for item in text:
            result = self._lexical_sentiment(item)
            sentiments.append(result["sentiment"])

        return np.array(sentiments, dtype=float)

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
                "sentiment": results,
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

    def _lexical_sentiment(self, text: str) -> Dict:
        lowered = text.lower()
        positive_terms = ("record", "profit", "profits", "gain", "gains", "surge", "bull", "partnership")
        negative_terms = ("uncertainty", "loss", "losses", "crash", "bear", "decline", "risk", "missed")
        positive = sum(1 for term in positive_terms if term in lowered)
        negative = sum(1 for term in negative_terms if term in lowered)
        score = 0.0
        if positive or negative:
            score = (positive - negative) / max(positive + negative, 1)
        return {
            "sentiment": float(max(-1.0, min(1.0, score))),
            "confidence": 0.8 if positive or negative else 0.5,
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'fingpt'):
                await self.fingpt.cleanup()
        except Exception as e:
            logger.error(f"Sentiment analyzer cleanup failed: {str(e)}")
