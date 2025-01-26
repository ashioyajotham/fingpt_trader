import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

import psutil

from models.portfolio.risk import MarketRegimeDetector, CircuitBreaker, MarketRegime

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService

logger = logging.getLogger(__name__)

class SystemMonitor(BaseService):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config = config or {}
        self.metrics: Dict[str, float] = {}
        self.service_status: Dict[str, str] = {}
        self.metrics_history: List[Dict] = []
        
        # Initialize risk monitoring
        risk_config = config.get('risk', {})
        self.regime_detector = MarketRegimeDetector()
        self.circuit_breaker = CircuitBreaker(risk_config.get('thresholds', {}))
        self.current_regime: Optional[MarketRegime] = None
        
        # Configure logging
        log_level = config.get('logging', {}).get('level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
        }

    async def _setup(self) -> None:
        self.check_interval = self.config.get("check_interval", 60)
        self.metrics_history = []

    async def collect_metrics(self) -> Dict:
        """Collect system metrics"""
        return {
            "timestamp": time.time(),
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
            "network": psutil.net_io_counters()._asdict(),
        }

    async def check_services(self, services: List[BaseService]) -> Dict:
        """Check health of all services"""
        status = {}
        for service in services:
            status[service.__class__.__name__] = {
                "status": service.get_status(),
                "running": service.is_running,
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

    async def update_metrics(self, market_data: Dict) -> None:
        """Update system metrics and risk status"""
        try:
            # Update market regime
            self.current_regime = self.regime_detector.detect_regime(market_data)
            
            # Check circuit breaker
            circuit_breaker_triggered = self.circuit_breaker.check_conditions(market_data)
            
            # Record metrics
            timestamp = datetime.now().isoformat()
            metrics = {
                'timestamp': timestamp,
                'market_regime': self.current_regime.value,
                'circuit_breaker': circuit_breaker_triggered,
                'volatility': market_data.get('volatility', 0),
                'liquidity': market_data.get('liquidity', 0),
                'correlation': market_data.get('correlation', 0)
            }
            
            self.metrics.update(metrics)
            self.metrics_history.append(metrics)
            
            # Log significant changes
            if circuit_breaker_triggered:
                logger.warning("Circuit breaker triggered")
            if self.current_regime in [MarketRegime.STRESS, MarketRegime.CRISIS]:
                logger.warning(f"Market regime changed to {self.current_regime.value}")
                
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
            raise
