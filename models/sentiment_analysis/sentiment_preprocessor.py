import re
from typing import List, Dict, Union
import numpy as np
from .fingpt_model.model import FinGPTModel

class SentimentPreprocessor:
    def __init__(self, max_length: int = 512, config_path: str = None):
        self.max_length = max_length
        self.model = FinGPTModel(config_path)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize financial text data"""
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove special characters but keep important financial symbols
        text = re.sub(r'[^\w\s$%\.+-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize financial symbols
        text = text.replace('$', ' $ ')
        text = text.replace('%', ' % ')
        
        return text.strip()
    
    def extract_financial_features(self, text: str) -> Dict[str, Union[bool, float]]:
        """Extract relevant financial indicators from text"""
        features = {
            'has_numbers': bool(re.search(r'\d', text)),
            'has_dollar': '$' in text,
            'has_percent': '%' in text,
            'has_price_mention': bool(re.search(r'\$\s*\d+\.?\d*', text)),
            'word_count': len(text.split()),
            'symbol_density': len(re.findall(r'[$%+-]', text)) / (len(text) + 1)
        }
        
        # Detect price changes
        price_changes = re.findall(r'(\+|-)?\s*\d+\.?\d*\s*%', text)
        if price_changes:
            features['max_price_change'] = max([float(pc.replace('%', '').strip()) 
                                              for pc in price_changes])
        else:
            features['max_price_change'] = 0.0
            
        return features
    
    def process_batch(self, texts: List[str], 
                     batch_size: int = 32) -> List[Dict[str, float]]:
        """Process a batch of texts and return sentiment scores with features"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts
            cleaned_texts = [self.clean_text(text) for text in batch]
            
            # Extract features
            features = [self.extract_financial_features(text) for text in cleaned_texts]
            
            # Get sentiment scores
            sentiments = self.model.analyze_sentiment(cleaned_texts)
            
            # Combine results
            for text, sentiment, feature in zip(cleaned_texts, sentiments, features):
                score = {
                    'positive': 1.0,
                    'neutral': 0.5,
                    'negative': 0.0
                }.get(sentiment.lower(), 0.5)
                
                # Adjust confidence based on features
                confidence = 1.0 if sentiment != 'neutral' else 0.5
                if feature['has_price_mention']:
                    confidence *= 1.2
                if feature['max_price_change'] > 0:
                    confidence *= 1.1
                    
                result = {
                    'original_text': text,
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': min(1.0, confidence),
                    'features': feature
                }
                
                results.append(result)
                
        return results
    
    def get_aggregate_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics for a collection of texts"""
        processed = self.process_batch(texts)
        
        scores = [p['score'] for p in processed]
        confidences = [p['confidence'] for p in processed]
        
        # Weight scores by confidence
        weighted_scores = np.array(scores) * np.array(confidences)
        
        return {
            'mean_sentiment': float(np.mean(weighted_scores)),
            'sentiment_std': float(np.std(scores)),
            'mean_confidence': float(np.mean(confidences)),
            'sample_size': len(processed),
            'bullish_ratio': len([s for s in scores if s > 0.5]) / len(scores)
        }
    
    def analyze_temporal_sentiment(self, texts: List[str], 
                                 timestamps: List[float]) -> Dict[str, float]:
        """Analyze sentiment changes over time"""
        if len(texts) != len(timestamps):
            raise ValueError("Number of texts and timestamps must match")
            
        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, texts), key=lambda x: x[0])
        _, sorted_texts = zip(*sorted_pairs)
        
        # Process in chronological order
        processed = self.process_batch(sorted_texts)
        scores = [p['score'] for p in processed]
        
        return {
            'sentiment_momentum': np.corrcoef(range(len(scores)), scores)[0,1],
            'recent_sentiment': np.mean(scores[-5:]),
            'sentiment_trend': np.polyfit(range(len(scores)), scores, 1)[0]
        }