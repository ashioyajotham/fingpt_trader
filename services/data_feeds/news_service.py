from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
import asyncio
from ...services.base_service import BaseService

class NewsService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        self.news_cache = {}
        self.cache_ttl = timedelta(minutes=15)
        self.rate_limit = 100  # calls per day
        self.calls_today = 0
        self.last_call = datetime.now()
        
    async def _setup(self) -> None:
        self.session = aiohttp.ClientSession()
        await self._validate_credentials()
        
    async def _cleanup(self) -> None:
        await self.session.close()
        
    async def get_news(self, query: str, limit: int = 10) -> List[Dict]:
        """Get latest news for query"""
        await self._check_rate_limit()
        
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit
        }
        
        cache_key = f"{query}_{limit}"
        if self._is_cache_valid(cache_key):
            return self.news_cache[cache_key]['data']
            
        try:
            async with self.session.get(f"{self.base_url}/everything", params=params) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                    
                data = await response.json()
                articles = data.get('articles', [])
                
                # Cache results
                self.news_cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': articles
                }
                
                return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            raise
            
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.news_cache:
            return False
        return datetime.now() - self.news_cache[key]['timestamp'] < self.cache_ttl
        
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        if datetime.now().date() > self.last_call.date():
            self.calls_today = 0
            
        if self.calls_today >= self.rate_limit:
            raise Exception("Daily rate limit exceeded")
            
        self.calls_today += 1
        self.last_call = datetime.now()
        
    async def _validate_credentials(self) -> None:
        """Validate API credentials"""
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not configured")