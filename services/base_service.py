from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from enum import Enum

class ServiceStatus(Enum):
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class BaseService(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def start(self) -> None:
        """Start service"""
        await self._setup()
        self.running = True
        
    async def stop(self) -> None:
        """Stop service"""
        self.running = False
        await self._cleanup()
    
    @abstractmethod
    async def _setup(self) -> None:
        """Initialize service resources"""
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup service resources"""
        pass
    
    async def _validate_config(self) -> None:
        """Validate service configuration"""
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if service is in running state"""
        return self.running
    
    def get_status(self) -> ServiceStatus:
        """Get current service status"""
        return self.status
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update service configuration"""
        self.config.update(config)