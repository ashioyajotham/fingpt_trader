from typing import Dict, List, Optional
from datetime import datetime
import aiohttp
import feedparser
from ...services.base_service import BaseService

class NewsService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.news_sources = []
        self.news_cache = []
        self.max_cache_size = 1000
        
    async def _setup(self) -> None:
        self.session = aiohttp.ClientSession()
        await self._init_sources()
        
    async def _cleanup(self) -> None:
        await self.session.close()
        
    async def get_latest_news(self, 
                             source: Optional[str] = None, 
                             limit: int = 10) -> List[Dict]:
        """Get latest news articles"""
        if source:
            return [n for n in self.news_cache if n['source'] == source][:limit]
        return self.news_cache[:limit]
        
    async def fetch_news_feed(self, source_url: str) -> List[Dict]:
        """Fetch news from RSS/API source"""
        async with self.session.get(source_url) as response:
            feed = feedparser.parse(await response.text())
            return self._parse_feed(feed)
            
    async def _init_sources(self) -> None:
        """Initialize news sources"""
        self.news_sources = self.config.get('news_sources', [])
        
    def _parse_feed(self, feed) -> List[Dict]:
        """Parse feed into standardized format"""
        articles = []
        for entry in feed.entries:
            articles.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', ''),
                'link': entry.get('link', ''),
                'source': feed.feed.get('title', '')
            })
        return articles