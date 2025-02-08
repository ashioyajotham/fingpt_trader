"""
FinGPT Trading System - Main Entry Point

"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from datetime import datetime

from utils.config import ConfigManager
from utils.logging import LogManager

# Initialize logging
LogManager({"log_dir": "logs", "level": "INFO"}).setup_basic_logging()
logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration path"""
        self.startup_time = datetime.now()
        self.is_running = False
        self.exchange_clients = {}
        
        # Load configuration
        self.config_manager = ConfigManager()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        if not self.config or 'exchanges' not in self.config:
            raise ValueError("Invalid trading configuration")
            
        logger.info("Trading system initialized with configuration")

    async def _setup_exchange(self, exchange_config: Dict) -> None:
        """Setup single exchange connection"""
        try:
            exchange_type = exchange_config.get('name', '').lower()
            
            if exchange_type == 'binance':
                from services.exchanges.binance import BinanceClient
                
                # Get validated credentials
                creds = self.config_manager.get_exchange_credentials('binance')
                
                # Create exchange client
                client_config = {
                    'api_key': creds['api_key'],
                    'api_secret': creds['api_secret'],
                    'test_mode': exchange_config.get('test_mode', True),
                    'options': exchange_config.get('options', {})
                }
                
                client = await BinanceClient.create(client_config)
                self.exchange_clients[exchange_type] = client
                logger.info(f"Successfully connected to {exchange_type}")
                
            else:
                logger.warning(f"Unsupported exchange type: {exchange_type}")
                
        except Exception as e:
            logger.error(f"Failed to setup {exchange_type}: {str(e)}")
            raise

    async def initialize(self) -> None:
        """Initialize all system components"""
        try:
            logger.info("Starting system initialization...")
            
            # Setup exchanges first
            for exchange_config in self.config['exchanges']:
                await self._setup_exchange(exchange_config)
                
            self.is_running = True
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.shutdown()
            raise

    async def run(self) -> None:
        """Main trading loop"""
        try:
            await self.initialize()
            
            while self.is_running:
                try:
                    # Trading loop implementation
                    await asyncio.sleep(
                        self.config.get('trading', {}).get('loop_interval', 60)
                    )
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    await asyncio.sleep(5)  # Brief pause before retry
                    
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Clean shutdown of all components"""
        self.is_running = False
        
        # Cleanup exchange connections
        for name, client in self.exchange_clients.items():
            try:
                await client.cleanup()
                logger.info(f"Cleaned up {name} connection")
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {str(e)}")

        logger.info("System shutdown complete")

if __name__ == "__main__":
    # Windows-specific settings
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create and run system
    system = TradingSystem("config/trading.yaml")
    
    try:
        asyncio.run(system.run())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)