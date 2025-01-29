import asyncio
import logging
import signal
from pathlib import Path
import sys
from typing import Dict
import yaml

from services.trading.robo_service import RoboService
from services.exchanges.binance import BinanceClient

logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.exchange = None
        self.robo_service = None

    async def startup(self):
        try:
            self.exchange = await BinanceClient.create(self.config['exchange'])
            self.robo_service = RoboService(self.config)
            await self.robo_service._setup()
            self.running = True
            logger.info("Trading system started successfully")
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            await self.shutdown()
            raise

    async def shutdown(self):
        self.running = False
        try:
            tasks = []
            
            if self.robo_service:
                tasks.append(self.robo_service.cleanup())
            if self.exchange:
                tasks.append(self.exchange.cleanup())
                
            # Wait for all cleanup tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Final wait to ensure all connections close
            await asyncio.sleep(0.5)
            
            logger.info("Trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise  # Re-raise to ensure proper process termination

    async def run(self):
        try:
            await self.startup()
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()

def load_config() -> Dict:
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        config = load_config()
        trading_system = TradingSystem(config)
        
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(trading_system.shutdown()))
        
        await trading_system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
