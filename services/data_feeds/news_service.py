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
        self.data_handlers = []  # Add handlers list
        
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
        """Start the news feed with periodic updates"""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        # Start background task for fetching news
        asyncio.create_task(self._periodic_news_fetch())
        
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

    async def subscribe(self, handler) -> None:
        """Register a data handler for news updates"""
        self.data_handlers.append(handler)
        logger.info(f"News handler registered: {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'}")

    async def _notify_handlers(self, news_item: Dict) -> None:
        """Notify all registered handlers of news updates"""
        for handler in self.data_handlers:
            try:
                await handler(news_item)
            except Exception as e:
                logger.error(f"News handler error: {e}")

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

    async def _periodic_news_fetch(self) -> None:
        """Periodically fetch news from API sources"""
        try:
            while self.running:
                await self._fetch_from_api()
                await asyncio.sleep(self.update_interval)  # Wait before next fetch
        except asyncio.CancelledError:
            logger.info("News fetching task cancelled")
        except Exception as e:
            logger.error(f"Error in news fetching task: {e}")

    async def _fetch_from_api(self) -> None:
        """Fetch news from external API"""
        if not hasattr(self, 'api_key') or not self.api_key:
            self.api_key = os.environ.get('CRYPTOPANIC_API_KEY')
            
        if not self.api_key:
            logger.error("No API key available for news service")
            return
            
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.api_key}&kind=news"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process each news item
                    for item in data.get('results', []):
                        news_item = {
                            'title': item.get('title', ''),
                            'content': item.get('body', ''),
                            'source': item.get('source', {}).get('title', 'Unknown'),
                            'timestamp': datetime.now(),
                            'url': item.get('url', '')
                        }
                        
                        # Add to cache and notify handlers
                        await self.process_news(news_item)
                        
                    logger.info(f"Fetched {len(data.get('results', []))} news items")
                else:
                    logger.error(f"News API error: {response.status} - {await response.text()}")
                    
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")

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

    def _calculate_relevance(self, news_item: Dict) -> float:
        """Calculate relevance score for news item (0.0 to 1.0)"""
        text = f"{news_item.get('title', '')} {news_item.get('content', '')}"
        
        # Initialize relevance score
        relevance = 0.0
        
        # Check for cryptocurrency mentions
        for pair, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    relevance += 0.2  # Increase relevance for each keyword match
                    break  # Only count each pair once
        
        # Check for recent publication (within last 24 hours)
        published_time = news_item.get('timestamp')
        if published_time and isinstance(published_time, datetime):
            hours_ago = (datetime.now() - published_time).total_seconds() / 3600
            if hours_ago < 24:
                relevance += 0.3  # More relevant if recent
        
        # Cap at 1.0 maximum
        return min(relevance, 1.0)

class CryptoPanicClient:
    """Client for CryptoPanic API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1"
        self.session = aiohttp.ClientSession()
    
    async def get_posts(self, currencies=None, kind="news", filter="hot", regions="en"):
        """Get posts from CryptoPanic API"""
        params = {
            "auth_token": self.api_key,
            "kind": kind,
            "filter": filter,
            "regions": regions,
            "public": "true"
        }
        
        if currencies:
            params["currencies"] = currencies
            
        try:
            url = f"{self.base_url}/posts/"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                else:
                    logger.error(f"CryptoPanic API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"CryptoPanic request failed: {str(e)}")
            return []
            
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
        # Clean up API clients
        if hasattr(self, 'cryptopanic_client'):
            await self.cryptopanic_client.close()
        if hasattr(self, 'news_api_client'):
            await self.news_api_client.close()
            
        self.cache.clear()

class NewsAPIClient:
    """Client for NewsAPI"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = aiohttp.ClientSession()
    
    async def get_top_headlines(self, q=None, category="business", language="en"):
        """Get top headlines from NewsAPI"""
        params = {
            "apiKey": self.api_key,
            "category": category,
            "language": language
        }
        
        if q:
            params["q"] = q
            
        try:
            url = f"{self.base_url}/top-headlines"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('articles', [])
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"NewsAPI request failed: {str(e)}")
            return []
            
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class NewsService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY")
        self.newsapi_key = os.getenv("NEWS_API_KEY")  # Fallback
        self.base_url = "https://cryptopanic.com/api/v1"
        self.fallback_url = "https://newsapi.org/v2"
        
        self.max_retries = 5  # Increased from 3
        self.retry_delay = 2  # Reduced from 5 seconds for faster retries
        self.request_timeout = 30  # Add explicit timeout
        
        # Add caching duration and size limits
        self.cache_ttl = timedelta(minutes=5)  # More frequent updates
        self.max_cache_size = 1000
        
        # Add debug logging for API responses
        self.debug_mode = True  # Enable verbose logging

        self.session = None
        self.cache = {}
        self.last_call = datetime.now()
        self.calls_today = 0
        self.rate_limit = self.config.get("rate_limits", {}).get("daily_limit", 100)
        self.update_interval = self.config.get("rate_limits", {}).get(
            "update_interval", 1
        )

    async def _setup(self) -> None:
        """Initialize news service"""
        if not self.cryptopanic_key:
            raise ValueError("CRYPTOPANIC_API_KEY not set")
        self.session = aiohttp.ClientSession()

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.cache.clear()

    async def setup(self, config=None):
        """Initialize the news service with API clients"""
        if config:
            self.config.update(config)
        
        try:
            # Initialize CryptoPanic client
            api_key = self.config.get('cryptopanic_api_key') or os.getenv('CRYPTOPANIC_API_KEY')
            if api_key:
                self.cryptopanic_client = CryptoPanicClient(api_key)
                
            # Initialize NewsAPI client as fallback
            news_api_key = self.config.get('news_api_key') or os.getenv('NEWS_API_KEY')
            if news_api_key:
                self.news_api_client = NewsAPIClient(news_api_key)
                
            logger.info("News data feed started")
        except Exception as e:
            logger.error(f"Failed to initialize news service: {str(e)}")
            raise

    async def get_news(self, query: str) -> List[Dict]:
        """Get latest news prioritizing CryptoPanic"""
        await self._check_rate_limit()
        
        # Map query to currency code
        currency_map = {
            'bitcoin cryptocurrency': 'BTC',
            'ethereum cryptocurrency': 'ETH',
            'binance coin': 'BNB'
        }
        currency = currency_map.get(query.lower()) or query.split()[0].upper()
        
        logger.info(f"Fetching news for currency: {currency}")

        # Try CryptoPanic first
        for attempt in range(self.max_retries):
            try:
                # Fix params format for CryptoPanic
                params = {
                    "auth_token": self.cryptopanic_key,
                    "currencies": currency,  # Use currency code instead of query
                    "kind": "news",  # Required parameter
                    "filter": "hot",
                    "regions": "en",  # English news only
                    "public": "true"
                }

                url = f"{self.base_url}/posts/"
                logger.debug(f"CryptoPanic request: {url} with params: {params}")

                async with self.session.get(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                    headers={"Accept": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Debug response
                        if self.debug_mode:
                            logger.info(f"CryptoPanic success - Results: {len(data.get('results', []))}")
                            
                        articles = self._process_cryptopanic_response(data)
                        if articles:
                            return articles
                            
                    else:
                        response_text = await response.text()
                        logger.warning(f"CryptoPanic error: Status {response.status}, Response: {response_text}")
                        
            except Exception as e:
                logger.error(f"CryptoPanic attempt {attempt+1} failed: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                
            if attempt == self.max_retries - 1:
                logger.info(f"Falling back to NewsAPI for {currency}")
                return await self._get_newsapi_fallback(query)
        
        return []

    def _process_cryptopanic_response(self, data: Dict) -> List[Dict]:
        """Process CryptoPanic response format"""
        articles = []
        results = data.get('results', [])
        
        if not results:
            logger.warning("No results in CryptoPanic response")
            return articles
            
        for item in results:
            try:
                article = {
                    'title': item.get('title', ''),
                    'description': item.get('text', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', {}).get('title', 'Unknown'),
                    'published_at': item.get('published_at'),
                    'currencies': [c.get('code', '') for c in item.get('currencies', [])]
                }
                
                # Only add if we have at least title or description
                if article['title'] or article['description']:
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
                
        logger.info(f"Processed {len(articles)} valid articles from CryptoPanic")
        return articles

    async def _get_newsapi_fallback(self, query: str) -> List[Dict]:
        """Fallback to NewsAPI"""
        try:
            params = {
                "q": query,
                "apiKey": self.newsapi_key,
                "sortBy": "publishedAt",
                "language": "en",
            }

            async with self.session.get(
                f"{self.fallback_url}/everything", params=params, timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("articles", [])
        except Exception as e:
            logger.error(f"NewsAPI fallback error: {e}")
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

    async def fetch_news(self, pairs: List[str]) -> List[Dict]:
        """Fetch news for multiple currency pairs"""
        news_items = []
        
        # Try CryptoPanic first
        if hasattr(self, 'cryptopanic_client') and self.cryptopanic_client:
            try:
                # Make max 5 attempts
                for attempt in range(1, 6):
                    try:
                        crypto_news = await self.cryptopanic_client.get_posts()
                        if crypto_news:
                            news_items.extend(crypto_news)
                            break
                    except Exception as e:
                        logger.error(f"CryptoPanic attempt {attempt} failed: {str(e)}")
                        await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"CryptoPanic API error: {str(e)}")
        
        # Try NewsAPI as fallback if needed
        if not news_items and hasattr(self, 'news_api_client') and self.news_api_client:
            try:
                # Construct query for crypto news
                query = " OR ".join([pair.replace("USDT", "") for pair in pairs])
                news_api_results = await self.news_api_client.get_top_headlines(q=query)
                if news_api_results:
                    news_items.extend(news_api_results)
            except Exception as e:
                logger.error(f"NewsAPI fallback error: {str(e)}")
        
        return news_items
        
    def is_relevant(self, item: Dict, pair: str) -> bool:
        """Check if a news item is relevant to a specific trading pair"""
        # Extract the currency part (e.g., "BTC" from "BTCUSDT")
        currency = pair.replace("USDT", "").lower()
        
        # Check title and content
        title = item.get('title', '').lower()
        body = item.get('body', '').lower()
        content = title + ' ' + body
        
        # Check if currency name or common names appear in content
        currency_keywords = {
            'btc': ['bitcoin', 'btc'],
            'eth': ['ethereum', 'eth'],
            'bnb': ['binance', 'bnb', 'binance coin']
        }
        
        # Get keywords for this currency
        keywords = currency_keywords.get(currency.lower(), [currency])
        
        # Check if any keywords appear in the content
        for keyword in keywords:
            if keyword in content:
                return True
                
        return False
