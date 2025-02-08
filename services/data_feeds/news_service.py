import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from services.base_service import BaseService

"""Real-time news and social media data feed service"""

import logging

logger = logging.getLogger(__name__)

class NewsDataFeed(BaseService):
    """Real-time news data feed handler"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.config = config or {}
        self.cache = []
        self.running = False
        self.update_interval = self.config.get('news_interval', 300)  # 5 minutes
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.session = None
        
        # Add news filters and processors
        self.filters = {
            'relevance': 0.5,  # Minimum relevance score
            'languages': ['en'],
            'sources': config.get('sources', [])
        }
        
        # Add crypto-specific keywords
        self.keywords = {
            'BTCUSDT': ['bitcoin', 'btc', 'crypto'],
            'ETHUSDT': ['ethereum', 'eth', 'defi'],
            # Add more pairs as needed
        }

    async def _setup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.running = True
            self.session = aiohttp.ClientSession()
            logger.info("News data feed setup complete")
        except Exception as e:
            logger.error(f"News data feed setup failed: {e}")
            raise

    async def _cleanup(self) -> None:
        """Required implementation of abstract method"""
        try:
            self.running = False
            if self.session:
                await self.session.close()
            self.cache.clear()
            logger.info("News data feed cleanup complete")
        except Exception as e:
            logger.error(f"News data feed cleanup failed: {e}")
            raise

    async def start(self) -> None:
        """Start the news feed"""
        self.running = True
        self.session = aiohttp.ClientSession()
        logger.info("News data feed started")

    async def stop(self) -> None:
        """Stop the news feed"""
        self.running = False
        if self.session:
            await self.session.close()
        self.cache.clear()
        logger.info("News data feed stopped")

    async def get_latest(self) -> List[Dict]:
        """Get latest news items"""
        return self.cache.copy()

    async def process_news(self, news_item: Dict) -> None:
        """Enhanced news processing with filtering"""
        # Check relevance
        if not self._check_relevance(news_item):
            return
            
        # Extract symbols
        symbols = self._extract_symbols(news_item)
        if not symbols:
            return
            
        # Process and cache
        processed_item = {
            'title': news_item.get('title', ''),
            'content': news_item.get('content', ''),
            'source': news_item.get('source', ''),
            'timestamp': datetime.now(),
            'symbols': symbols,
            'relevance': self._calculate_relevance(news_item)
        }
        
        self.cache.append(processed_item)
        await self._notify_handlers(processed_item)
        
        # Maintain cache size
        if len(self.cache) > self.max_cache_size:
            self.cache.pop(0)

    def _check_relevance(self, news_item: Dict) -> bool:
        """Check if news item is relevant"""
        text = f"{news_item.get('title', '')} {news_item.get('content', '')}"
        
        # Check keywords
        for pair, keywords in self.keywords.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return True
                
        return False

    def _extract_symbols(self, news_item: Dict) -> List[str]:
        """Extract trading pairs from news"""
        text = f"{news_item.get('title', '')} {news_item.get('content', '')}"
        symbols = []
        
        for pair, keywords in self.keywords.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                symbols.append(pair)
                
        return symbols

class NewsService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.api_key = os.getenv("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2"
        self.fallback_urls = [
            "https://cryptopanic.com/api/v1",
            "https://api.alternative.me/v2",
        ]
        self.max_retries = 3
        self.retry_delay = 5
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        self.last_call = datetime.now()
        self.calls_today = 0
        self.rate_limit = self.config.get("rate_limits", {}).get("daily_limit", 100)
        self.update_interval = self.config.get("rate_limits", {}).get(
            "update_interval", 1
        )

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
                    "language": "en",
                }

                async with self.session.get(
                    f"{self.base_url}/everything", params=params, timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("articles", [])
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
