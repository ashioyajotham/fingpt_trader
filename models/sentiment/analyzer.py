from typing import List, Dict, Union
from datetime import datetime
import pandas as pd
import numpy as np
from ..llm.fingpt import FinGPT
from ..llm.utils.tokenizer import TokenizerConfig

class SentimentAnalyzer:
    def __init__(self, model_config: Dict):
        self.model = FinGPT(model_config)
        self.model.load_model()
        self.batch_size = model_config.get('batch_size', 16)
        
    def analyze_text(self, text: Union[str, List[str]]) -> Dict:
        """Analyze sentiment of single text or list of texts"""
        if isinstance(text, str):
            text = [text]
            
        sentiments = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i:i + self.batch_size]
            batch_sentiments = self.model.predict_sentiment(batch)
            sentiments.extend(batch_sentiments)
            
        return {
            'sentiments': sentiments,
            'timestamp': datetime.now().isoformat(),
            'summary': self._summarize_sentiments(sentiments)
        }
        
    def analyze_news_feed(self, news_items: List[Dict]) -> pd.DataFrame:
        """Analyze sentiment for news feed"""
        texts = [item['title'] + ' ' + item.get('description', '') 
                for item in news_items]
        
        results = self.analyze_text(texts)
        
        df = pd.DataFrame({
            'timestamp': [item.get('timestamp') for item in news_items],
            'text': texts,
            'sentiment': results['sentiments'],
            'source': [item.get('source') for item in news_items]
        })
        
        return df
    
    def _summarize_sentiments(self, sentiments: List[str]) -> Dict:
        """Generate sentiment summary statistics"""
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        scores = [sentiment_map[s] for s in sentiments]
        
        return {
            'mean_score': np.mean(scores),
            'sentiment_counts': {
                label: sentiments.count(label)
                for label in sentiment_map.keys()
            },
            'majority_sentiment': max(set(sentiments), key=sentiments.count)
        }
