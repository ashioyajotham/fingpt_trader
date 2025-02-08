import asyncio
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class RetryHandler:
    """Handles retrying failed operations"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff

    async def retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retries"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                delay = self.delay * (self.backoff ** attempt)
                
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                
        logger.error(f"Operation failed after {self.max_retries} attempts: {last_error}")
        raise last_error
