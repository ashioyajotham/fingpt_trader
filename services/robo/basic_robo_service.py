from .robo_service import RoboService

class BasicRoboService(RoboService):
    """Basic implementation of robo-advisor service"""
    
    async def _cleanup(self) -> None:
        """Implement resource cleanup"""
        # Cleanup resources
        pass
        
    async def start(self) -> None:
        """Start service implementation"""
        # Implementation
        pass
        
    async def stop(self) -> None:
        """Stop service implementation"""
        await self._cleanup()