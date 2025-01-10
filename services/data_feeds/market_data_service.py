from typing import Dict, List, Optional, Union
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from ...services.base_service import BaseService
import aiohttp
import os
from ...utils.config import ConfigManager

class MarketDataService(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config = config or {}
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        self.cache = {}
        self.last_call = datetime.now()
        self.call_limit = 5  # calls per minute
        self.data_cache = {}
        self.subscribers = []
        self.rate_limits = {
            'api_calls_per_minute': 60,
            'max_historical_days': 365
        }
        
    async def _setup(self) -> None:
        self.session = aiohttp.ClientSession()  # Initialize API session
        self.last_update = datetime.now()
        await self._init_cache()
        
    async def _cleanup(self) -> None:
        if self.session:
            await self.session.close()
            
    async def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time market data from Alpha Vantage"""
        await self._check_rate_limit()
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                    
                data = await response.json()
                quote = data.get("Global Quote", {})
                
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'timestamp': datetime.now(),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', '0').strip('%'))
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            raise
        
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch historical daily data from Alpha Vantage"""
        await self._check_rate_limit()
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                    
                data = await response.json()
                time_series = data.get("Time Series (Daily)", {})
                
                # Transform to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                
                # Rename columns
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Convert types
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col])
                df['volume'] = pd.to_numeric(df['volume'], dtype='int64')
                
                # Filter date range
                df = df.loc[start_date:end_date] if end_date else df.loc[start_date:]
                
                # Cache results
                self.cache[f"{symbol}_{start_date}_{end_date}"] = df
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
        
    async def subscribe(self, callback, symbols: List[str]) -> None:
        """Subscribe to real-time updates"""
        self.subscribers.append((callback, symbols))
        
    async def _init_cache(self) -> None:
        """Initialize data cache"""
        self.data_cache = {}
        
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        current_time = datetime.now()
        if (current_time - self.last_update).seconds < 1:
            await asyncio.sleep(1)
        self.last_update = current_time