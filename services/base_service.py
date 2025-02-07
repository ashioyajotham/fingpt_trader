import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BaseService(ABC):
    """
    Abstract base class for all services in the trading system.
    
    Provides consistent initialization and cleanup patterns.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.initialized = False

    async def initialize(self) -> None:
        """Public initialization method"""
        if self.initialized:
            return
            
        try:
            await self._setup()
            self.initialized = True
            logger.info(f"{self.__class__.__name__} initialized")
        except Exception as e:
            logger.error(f"{self.__class__.__name__} initialization failed: {str(e)}")
            raise

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
        """Implementation-specific setup logic"""
        pass

    @abstractmethod
    async def _cleanup(self) -> None:
        """Implementation-specific cleanup logic"""
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
