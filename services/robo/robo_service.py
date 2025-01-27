from abc import ABC, abstractmethod
from typing import Dict, Any

class RoboService(ABC):
    """Base class for robo-advisor services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup resources"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start robo-advisor service"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop robo-advisor service"""
        # Ensure cleanup is called
        await self._cleanup()