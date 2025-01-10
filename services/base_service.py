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
        self.status = ServiceStatus.INITIALIZED
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def start(self) -> None:
        """Initialize and start the service"""
        self.status = ServiceStatus.STARTING
        try:
            await self._validate_config()
            await self._setup()
            self.status = ServiceStatus.RUNNING
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Failed to start service: {str(e)}")
            raise
    
    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the service"""
        self.status = ServiceStatus.STOPPING
        try:
            await self._cleanup()
            self.status = ServiceStatus.STOPPED
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Failed to stop service: {str(e)}")
            raise
    
    @abstractmethod
    async def _setup(self) -> None:
        """Service-specific setup logic"""
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """Service-specific cleanup logic"""
        pass
    
    async def _validate_config(self) -> None:
        """Validate service configuration"""
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if service is in running state"""
        return self.status == ServiceStatus.RUNNING
    
    def get_status(self) -> ServiceStatus:
        """Get current service status"""
        return self.status
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update service configuration"""
        self.config.update(config)