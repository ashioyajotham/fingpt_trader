import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging
from web3 import Web3
import asyncio
import requests
from aiohttp import ClientSession, TCPConnector
from web3.providers import AsyncHTTPProvider

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)

from models.portfolio.rebalancing import Portfolio
from services.base_service import BaseService
from models.client.profile import MockClientProfile

logger = logging.getLogger(__name__)

class RoboService(BaseService):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.web3_session = None
        self.connector = None  # Add explicit connector reference
        self.w3 = None
        self.mempool_monitor = None
        self.infura_url = config.get('infura_url', 'https://mainnet.infura.io/v3/your-project-id')

    async def _setup(self):
        """Required implementation of abstract _setup method"""
        try:
            # Create and store connector reference
            self.connector = TCPConnector(force_close=True)
            self.web3_session = ClientSession(connector=self.connector)
            
            # Initialize Web3 with session
            self.w3 = Web3(AsyncHTTPProvider(
                self.infura_url,
                session=self.web3_session
            ))
            
            # Start mempool monitoring
            self.mempool_monitor = asyncio.create_task(self._setup_mempool_monitor())
            logger.info("RoboService setup complete")
            
        except Exception as e:
            logger.error(f"RoboService setup failed: {e}")
            await self._cleanup()  # Ensure cleanup on failure
            raise

    async def _cleanup(self):
        """Required implementation of abstract _cleanup method"""
        try:
            # Cancel mempool monitoring first
            if self.mempool_monitor:
                self.mempool_monitor.cancel()
                try:
                    await self.mempool_monitor
                except asyncio.CancelledError:
                    pass
                self.mempool_monitor = None

            # Close session and connector in correct order
            if self.web3_session and not self.web3_session.closed:
                await self.web3_session.close()
                self.web3_session = None

            if self.connector and not self.connector.closed:
                await self.connector.close()
                self.connector = None
                
            self.w3 = None
            logger.info("RoboService cleaned up")
            
        except Exception as e:
            logger.error(f"RoboService cleanup failed: {e}")

    async def _setup_mempool_monitor(self):
        """Setup mempool monitoring"""
        try:
            logger.info("Starting mempool monitoring...")  # Add this log
            while True:
                try:
                    # Get latest block
                    block = await self.w3.eth.get_block('latest')
                    
                    # Convert AttributeDict to regular dict
                    block_dict = dict(block)
                    
                    # Add block info logging
                    logger.debug(f"Processing block {block_dict.get('number', 'unknown')}")
                    
                    # Process transactions
                    if block_dict and 'transactions' in block_dict:
                        tx_count = len(block_dict['transactions'])
                        logger.info(f"Found {tx_count} transactions in latest block")
                        
                        for tx_hash in block_dict['transactions']:
                            try:
                                # Get transaction details
                                tx = await self.w3.eth.get_transaction(tx_hash)
                                tx_dict = dict(tx) if tx else None
                                
                                if tx_dict:
                                    await self._process_transaction(tx_dict)
                            except Exception as tx_error:
                                logger.error(f"Error processing transaction {tx_hash}: {tx_error}")
                                continue
                    
                    # Wait before next check
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("Mempool monitoring cancelled")
                    break
                except Exception as e:
                    logger.error(f"Mempool monitoring error: {str(e)}")
                    await asyncio.sleep(5)  # Back off on error
                    
        except asyncio.CancelledError:
            logger.info("Mempool monitor setup cancelled")
            raise
        except Exception as e:
            logger.error(f"Failed to setup mempool monitor: {e}")
            raise

    async def _process_transaction(self, tx: Dict):
        """Process a single transaction"""
        try:
            # Your transaction processing logic here
            pass
        except Exception as e:
            logger.error(f"Transaction processing error: {e}")