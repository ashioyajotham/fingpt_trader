import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import asyncio
import aiohttp
from datetime import datetime, timedelta

class MarketDataService:
    def __init__(self, config: Dict):
        self.cache = {}
        self.cache_duration = config.get('cache_duration', 300)  # 5 minutes default
        self.news_sources = config.get('news_sources', ['reuters', 'bloomberg', 'wsj'])
        
    async def get_market_data(self, 
                            symbols: List[str], 
                            interval: str = '1d',
                            period: str = '1mo') -> Dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols"""
        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_symbol_data(session, symbol, interval, period)
                for symbol in symbols
            ]
            completed = await asyncio.gather(*tasks)
            
            for symbol, data in zip(symbols, completed):
                if data is not None:
                    results[symbol] = data
                    
        return results
    
    async def _fetch_symbol_data(self, 
                               session: aiohttp.ClientSession,
                               symbol: str,
                               interval: str,
                               period: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol"""
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                return data
                
        try:
            # Using yfinance for demonstration
            data = yf.download(symbol, interval=interval, period=period)
            self.cache[cache_key] = (datetime.now(), data)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    async def get_news_data(self, symbols: List[str], 
                           lookback_days: int = 7) -> Dict[str, List[Dict]]:
        """Fetch news data for symbols"""
        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_symbol_news(session, symbol, lookback_days)
                for symbol in symbols
            ]
            completed = await asyncio.gather(*tasks)
            
            for symbol, news in zip(symbols, completed):
                if news:
                    results[symbol] = news
                    
        return results
    
    async def _fetch_symbol_news(self,
                               session: aiohttp.ClientSession,
                               symbol: str,
                               lookback_days: int) -> List[Dict]:
        """Fetch news for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Filter and format news
            formatted_news = []
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            for article in news:
                if datetime.fromtimestamp(article['providerPublishTime']) >= cutoff_date:
                    formatted_news.append({
                        'title': article['title'],
                        'timestamp': article['providerPublishTime'],
                        'source': article.get('source', ''),
                        'url': article.get('link', '')
                    })
                    
            return formatted_news
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return []