import re
from typing import Dict, List, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class SentimentPreprocessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Financial terms to preserve
        self.financial_terms = {
            "bullish",
            "bearish",
            "long",
            "short",
            "buy",
            "sell",
            "calls",
            "puts",
            "margin",
            "dividend",
            "earnings",
        }

        # Add financial terms to exceptions
        self.stop_words -= self.financial_terms

    def preprocess(self, text: str) -> List[str]:
        """Main preprocessing pipeline"""
        # Clean text
        text = self._clean_text(text)

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def _clean_text(self, text: str) -> str:
        """Clean raw text"""
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove special characters
        text = re.sub(r"[^\w\s]", "", text)

        # Remove numbers but preserve those in tickers
        text = re.sub(r"(?<![A-Z])\d+", "", text)

        # Handle stock tickers ($AAPL)
        text = re.sub(r"\$([A-Z]+)", r"\1", text)

        return text.strip()

    def extract_features(self, tokens: List[str]) -> Dict:
        """Extract features from tokens"""
        return {
            "token_count": len(tokens),
            "financial_terms": sum(1 for t in tokens if t in self.financial_terms),
            "unique_tokens": len(set(tokens)),
        }
