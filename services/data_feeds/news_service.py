from ..base_service import BaseService
from typing import Dict, List, Optional
import aiohttp
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import re

class NewsService(BaseService):
    def _validate_config(self) -> None:
        required = ['api_keys', 'news_sources']
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def initialize(self) -> None:
        self.cache = {}
        self.cache_duration = self.config.get('cache_duration', 300)  # 5 minutes
        self.session = None
        self.source_apis = {
            'reuters': self._fetch_reuters,
            'bloomberg': self._fetch_bloomberg,
            'finviz': self._fetch_finviz,
            'alphavantage': self._fetch_alphavantage
        }

    async def shutdown(self) -> None:
        if self.session:
            await self.session.close()

    async def _ensure_session(self) -> None:
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def get_company_news(self, 
                             symbol: str, 
                             lookback_days: int = 7,
                             sources: Optional[List[str]] = None) -> List[Dict]:
        """Fetch and aggregate news from multiple sources for a company"""
        await self._ensure_session()
        sources = sources or self.config['news_sources']
        
        cache_key = f"{symbol}_{lookback_days}_{'-'.join(sources)}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                return data

        tasks = []
        for source in sources:
            if source in self.source_apis:
                tasks.append(self.source_apis[source](symbol, lookback_days))

        results = await asyncio.gather(*tasks)
        all_news = []
        for news_items in results:
            all_news.extend(news_items)

        # Sort by timestamp and deduplicate
        all_news.sort(key=lambda x: x['timestamp'], reverse=True)
        unique_news = self._deduplicate_news(all_news)
        
        self.cache[cache_key] = (datetime.now(), unique_news)
        return unique_news

    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news based on title similarity"""
        seen_titles = set()
        unique_news = []
        
        for item in news_items:
            title = self._normalize_title(item['title'])
            if title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(item)
                
        return unique_news

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        return re.sub(r'[^\w\s]', '', title.lower())

    async def get_sector_news(self, 
                            sector: str, 
                            lookback_days: int = 7) -> List[Dict]:
        """Fetch news related to a specific sector"""
        # Implementation specific to sector news
        pass

    async def get_market_news(self, 
                            lookback_days: int = 7) -> List[Dict]:
        """Fetch general market news"""
        # Implementation for market-wide news
        pass

    async def _fetch_reuters(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Reuters"""
        # Implementation for Reuters API
        pass

    async def _fetch_bloomberg(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Bloomberg"""
        # Implementation for Bloomberg API
        pass

    async def _fetch_finviz(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Finviz"""
        # Implementation for Finviz API
        pass

    async def _fetch_alphavantage(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        # Implementation for Alpha Vantage API
        pass
