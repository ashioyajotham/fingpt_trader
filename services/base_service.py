from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseService(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self.initialize()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate service configuration"""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize service resources"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup service resources"""
        pass
