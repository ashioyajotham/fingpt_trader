import re
import string
from typing import List, Optional
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add finance-specific stop words
        self.stop_words.update(['stock', 'market', 'price', 'trading'])
        
        # Common financial symbols/tickers cleanup
        self.ticker_pattern = re.compile(r'\$[A-Z]+')
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove stock tickers ($AAPL)
        text = self.ticker_pattern.sub('', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def preprocess(self, text: str) -> List[str]:
        """Full preprocessing pipeline"""
        # Clean
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        return tokens
    
    def batch_preprocess(self, texts: List[str]) -> List[List[str]]:
        """Process multiple texts"""
        return [self.preprocess(text) for text in texts]
    
    def preprocess_df(self, df: pd.DataFrame, text_col: str, 
                     processed_col: Optional[str] = None) -> pd.DataFrame:
        """Preprocess text column in DataFrame"""
        if processed_col is None:
            processed_col = f'{text_col}_processed'
            
        df[processed_col] = df[text_col].apply(self.preprocess)
        return df