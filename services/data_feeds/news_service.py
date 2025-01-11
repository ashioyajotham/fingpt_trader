from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
import os

from pathlib import Path
import sys
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.base_service import BaseService

class NewsService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        self.fallback_urls = [
            "https://cryptopanic.com/api/v1",
            "https://api.alternative.me/v2"
        ]
        self.max_retries = 3
        self.retry_delay = 5
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        self.last_call = datetime.now()
        self.calls_today = 0
        self.rate_limit = self.config.get('rate_limits', {}).get('daily_limit', 100)
        self.update_interval = self.config.get('rate_limits', {}).get('update_interval', 1)

    async def _setup(self) -> None:
        """Initialize news service"""
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not set")
        self.session = aiohttp.ClientSession()

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.cache.clear()

    async def get_news(self, query: str) -> List[Dict]:
        """Get latest news for query"""
        await self._check_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                params = {
                    "q": query,
                    "apiKey": self.api_key,
                    "sortBy": "publishedAt",
                    "language": "en"
                }
                
                async with self.session.get(f"{self.base_url}/everything", params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Error fetching news, trying fallback sources")
                    return await self._try_fallback_sources(query)
                await asyncio.sleep(self.retry_delay)
        return []

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        current_time = datetime.now()
        
        # Reset daily counter
        if current_time.date() > self.last_call.date():
            self.calls_today = 0
            
        # Check daily limit
        if self.calls_today >= self.rate_limit:
            raise Exception("Daily API call limit exceeded")
            
        # Check interval
        time_diff = (current_time - self.last_call).total_seconds()
        if time_diff < self.update_interval:
            await asyncio.sleep(self.update_interval - time_diff)
            
        self.calls_today += 1
        self.last_call = current_time
        
    async def _validate_credentials(self) -> None:
        """Validate API credentials"""
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not configured")