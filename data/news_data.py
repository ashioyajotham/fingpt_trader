from datetime import datetime
from typing import Dict, List

import aiohttp


class NewsDataSource:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.cache_duration = 300  # 5 minutes

    async def get_news(self, symbols: List[str]) -> List[Dict]:
        params = {
            "q": " OR ".join(symbols),
            "apiKey": self.api_key,
            "sortBy": "publishedAt",
        }
        async with self.session.get(
            "https://newsapi.org/v2/everything", params=params
        ) as response:
            data = await response.json()
            return data["articles"]
