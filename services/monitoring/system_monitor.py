from typing import Dict, List
import psutil
import time

import sys
from pathlib import Path
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService

class SystemMonitor(BaseService):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.metrics = {}
        self.service_status = {}
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }

    async def _setup(self) -> None:
        self.check_interval = self.config.get('check_interval', 60)
        self.metrics_history = []
        
    async def collect_metrics(self) -> Dict:
        """Collect system metrics"""
        return {
            'timestamp': time.time(),
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'network': psutil.net_io_counters()._asdict()
        }
        
    async def check_services(self, services: List[BaseService]) -> Dict:
        """Check health of all services"""
        status = {}
        for service in services:
            status[service.__class__.__name__] = {
                'status': service.get_status(),
                'running': service.is_running
            }
        return status
        
    async def check_thresholds(self) -> List[str]:
        """Check if metrics exceed thresholds"""
        alerts = []
        metrics = await self.collect_metrics()
        
        for metric, threshold in self.thresholds.items():
            if metrics.get(metric, 0) > threshold:
                alerts.append(f"{metric} exceeded threshold: {metrics[metric]}")
        
        return alerts