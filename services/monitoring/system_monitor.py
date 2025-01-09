from ..base_service import BaseService
from typing import Dict, List
import psutil
import logging
from datetime import datetime
import asyncio

class SystemMonitor(BaseService):
    def _validate_config(self) -> None:
        self.log_path = self.config.get('log_path', 'logs/system.log')
        self.check_interval = self.config.get('check_interval', 60)

    def initialize(self) -> None:
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.metrics = {}
        self.running = False

    async def shutdown(self) -> None:
        self.running = False
        await asyncio.sleep(0.1)  # Allow final metrics to be collected

    async def start_monitoring(self) -> None:
        self.running = True
        while self.running:
            metrics = self.collect_metrics()
            self.metrics = metrics
            logging.info(f"System metrics: {metrics}")
            await asyncio.sleep(self.check_interval)

    def collect_metrics(self) -> Dict:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }

    def get_latest_metrics(self) -> Dict:
        return self.metrics