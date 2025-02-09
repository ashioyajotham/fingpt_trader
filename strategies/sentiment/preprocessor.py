import re
from typing import Dict, List, Set
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)

class SentimentPreprocessor:
    """Text preprocessor optimized for financial sentiment analysis"""
    
    def __init__(self):
        try:
            # Download required NLTK data silently
            for resource in ['punkt', 'stopwords', 'wordnet']:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.error(f"Failed to download NLTK resource {resource}: {e}")

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))

            # Extended financial terms to preserve
            self.financial_terms = {
                # Market sentiment
                "bullish", "bearish", "neutral", "rally", "dump", "crash", "moon",
                # Trading terms
                "long", "short", "buy", "sell", "hold", "margin", "leverage",
                # Analysis terms
                "support", "resistance", "breakout", "breakdown", "trend",
                # Market events
                "ath", "atl", "dip", "correction", "accumulation", "distribution",
                # Crypto specific
                "btc", "eth", "blockchain", "defi", "nft", "wallet", "mining",
                # Technical terms
                "rsi", "macd", "volume", "volatility", "momentum"
            }

            # Remove financial terms from stopwords
            self.stop_words -= self.financial_terms
            
            logger.info("SentimentPreprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SentimentPreprocessor: {e}")
            raise

    def preprocess(self, text: str, remove_urls: bool = True) -> List[str]:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text input
            remove_urls: Whether to remove URLs
            
        Returns:
            List of processed tokens
        """
        try:
            # Basic cleaning
            text = self._clean_text(text, remove_urls)
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            processed_tokens = []
            for sentence in sentences:
                # Tokenize
                tokens = word_tokenize(sentence.lower())
                
                # Remove stop words but preserve financial terms
                tokens = [
                    t for t in tokens 
                    if t in self.financial_terms or t not in self.stop_words
                ]
                
                # Lemmatize
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                
                processed_tokens.extend(tokens)
            
            return processed_tokens
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return []

    def _clean_text(self, text: str, remove_urls: bool = True) -> str:
        """Clean and normalize text"""
        try:
            # Remove URLs if specified
            if remove_urls:
                text = re.sub(r"http\S+|www\S+", "", text)

            # Normalize whitespace
            text = " ".join(text.split())
            
            # Handle crypto tickers ($BTC, $ETH)
            text = re.sub(r'\$([A-Z]{2,})', r'\1', text)
            
            # Remove special characters but preserve % and $
            text = re.sub(r'[^\w\s%$+-]', ' ', text)
            
            # Normalize numbers with units
            text = re.sub(r'(\d+)k\b', r'\1000', text)
            text = re.sub(r'(\d+)m\b', r'\1000000', text)
            
            # Handle percentages
            text = re.sub(r'(\d+)%', r'\1 percent', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text

    def extract_features(self, tokens: List[str]) -> Dict:
        """Extract features from processed tokens"""
        try:
            # Basic features
            features = {
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens)),
                "financial_terms": sum(1 for t in tokens if t in self.financial_terms),
                "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0
            }
            
            # Sentiment term counts
            sentiment_terms = {
                "positive": sum(1 for t in tokens if t in {"bullish", "buy", "long"}),
                "negative": sum(1 for t in tokens if t in {"bearish", "sell", "short"}),
                "neutral": sum(1 for t in tokens if t in {"hold", "neutral"})
            }
            features.update(sentiment_terms)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

    def get_financial_impact(self, tokens: List[str]) -> float:
        """Calculate financial impact score"""
        try:
            # Count impactful terms
            impact_terms = {
                "high": {"surge", "soar", "jump", "rally", "breakout"},
                "medium": {"rise", "gain", "increase", "improve"},
                "low": {"edge", "drift", "slight"},
                "negative": {"drop", "fall", "decline", "slide", "dump"}
            }
            
            # Calculate weighted impact
            weights = {"high": 1.0, "medium": 0.6, "low": 0.3, "negative": -1.0}
            impact = 0.0
            
            for term in tokens:
                for level, terms in impact_terms.items():
                    if term in terms:
                        impact += weights[level]
                        break
                        
            return max(min(impact, 1.0), -1.0)  # Normalize to [-1, 1]
            
        except Exception as e:
            logger.error(f"Impact calculation failed: {e}")
            return 0.0
