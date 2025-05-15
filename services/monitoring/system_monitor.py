import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import pandas as pd

import psutil

from models.portfolio.risk import MarketRegimeDetector, CircuitBreaker, MarketRegime

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from services.base_service import BaseService

logger = logging.getLogger(__name__)

class SystemMonitor(BaseService):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.price_history = {}  # Store price history for rapid move detection
        self.regime_detector = MarketRegimeDetector(config.get('regime_detector', {}))
        self.circuit_breaker = CircuitBreaker(config.get('circuit_breaker', {}))
        self.config = config or {}
        self.metrics: Dict[str, float] = {}
        self.service_status: Dict[str, str] = {}
        self.metrics_history: List[Dict] = []
        
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

    async def check_health(self):
        """
        Check system health metrics including resource usage and market conditions
        
        Returns:
            dict: Health status containing metrics and status evaluation
        """
        health_data = {
            'status': 'healthy',
            'metrics': {},
            'critical_metrics': []
        }
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            health_data['metrics']['cpu_percent'] = cpu_percent
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            health_data['metrics']['memory_percent'] = memory_percent
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            health_data['metrics']['disk_percent'] = disk_percent
            
            # Update metrics history
            self.metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add to history
            self.metrics_history.append(self.metrics.copy())
            if len(self.metrics_history) > 100:  # Limit history size
                self.metrics_history = self.metrics_history[-100:]
            
            # Evaluate thresholds
            cpu_threshold = self.thresholds.get('cpu_percent', 80.0)
            memory_threshold = self.thresholds.get('memory_percent', 85.0)
            disk_threshold = self.thresholds.get('disk_percent', 90.0)
            
            # Check for warning conditions
            warning_metrics = []
            critical_metrics = []
            
            if cpu_percent > cpu_threshold:
                critical_metrics.append(f"CPU: {cpu_percent}%")
            elif cpu_percent > cpu_threshold * 0.8:
                warning_metrics.append(f"CPU: {cpu_percent}%")
                
            if memory_percent > memory_threshold:
                critical_metrics.append(f"Memory: {memory_percent}%")
            elif memory_percent > memory_threshold * 0.8:
                warning_metrics.append(f"Memory: {memory_percent}%")
                
            if disk_percent > disk_threshold:
                critical_metrics.append(f"Disk: {disk_percent}%")
            elif disk_percent > disk_threshold * 0.8:
                warning_metrics.append(f"Disk: {disk_percent}%")
            
            # Update status based on metrics
            if critical_metrics:
                health_data['status'] = 'critical'
                health_data['critical_metrics'] = critical_metrics
                logger.warning(f"System resources critical: {', '.join(critical_metrics)}")
            elif warning_metrics:
                health_data['status'] = 'warning'
                health_data['warning_metrics'] = warning_metrics
                logger.info(f"System resources warning: {', '.join(warning_metrics)}")
            
            # Add service status checks if configured
            if self.config.get('check_services', False):
                services = self.check_service_health()
                health_data['services'] = services
                
                # Check for failed services
                failed_services = [name for name, status in services.items() if status != 'healthy']
                if failed_services:
                    health_data['status'] = 'critical'
                    health_data['critical_metrics'].append(f"Failed services: {', '.join(failed_services)}")
                    logger.warning(f"Services unhealthy: {failed_services}")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            health_data['status'] = 'error'
            health_data['error'] = str(e)
        
        return health_data

    def check_service_health(self):
        """Check the health of registered services"""
        service_status = {}
        
        # Example services to check - these should be configurable
        # In a real implementation, you would ping these services
        services = self.config.get('services', [])
        for service_name in services:
            try:
                # Placeholder for actual service check
                # This would typically involve a ping or heartbeat check
                service_status[service_name] = self.service_status.get(service_name, 'unknown')
            except Exception as e:
                logger.error(f"Error checking service {service_name}: {e}")
                service_status[service_name] = 'error'
        
        return service_status

    def detect_rapid_moves(self, market_data):
        """
        Detect rapid price movements that may indicate market volatility
        
        Args:
            market_data (dict): Dictionary of market data by symbol
            
        Returns:
            dict: Symbols with rapid movements and their metrics
        """
        rapid_moves = {}
        
        # Configuration
        threshold_pct = self.config.get('rapid_move_threshold', 3.0)  # Default 3%
        window_minutes = self.config.get('price_window_minutes', 5)   # Default 5 minutes
        
        current_time = datetime.now()
        
        # Process each symbol
        for symbol, data in market_data.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            # Store current price with timestamp
            if 'price' in data and data['price']:
                price_point = {
                    'price': float(data['price']),
                    'timestamp': current_time
                }
                
                # Add to history
                self.price_history[symbol].append(price_point)
                
                # Clean old data (older than window)
                cutoff_time = current_time - pd.Timedelta(minutes=window_minutes)
                self.price_history[symbol] = [
                    p for p in self.price_history[symbol] 
                    if p['timestamp'] >= cutoff_time
                ]
                
                # Check for rapid moves if we have enough data
                if len(self.price_history[symbol]) >= 2:
                    oldest_point = self.price_history[symbol][0]
                    newest_point = self.price_history[symbol][-1]
                    
                    # Calculate price change
                    price_change_pct = ((newest_point['price'] - oldest_point['price']) / 
                                        oldest_point['price']) * 100
                    
                    # Check if change exceeds threshold
                    if abs(price_change_pct) >= threshold_pct:
                        direction = "up" if price_change_pct > 0 else "down"
                        duration_seconds = (newest_point['timestamp'] - 
                                           oldest_point['timestamp']).total_seconds()
                        
                        rapid_moves[symbol] = {
                            'change_pct': price_change_pct,
                            'direction': direction,
                            'window_seconds': duration_seconds,
                            'start_price': oldest_point['price'],
                            'current_price': newest_point['price']
                        }
                        
                        # Log the rapid move
                        logger.warning(f"Rapid price move detected in {symbol}: "
                                      f"{price_change_pct:.2f}% {direction} "
                                      f"in {duration_seconds:.1f} seconds")
        
        return rapid_moves

    async def register_service(self, service_name, initial_status='healthy'):
        """Register a service for health monitoring"""
        self.service_status[service_name] = initial_status
        logger.info(f"Service registered for monitoring: {service_name}")

    async def update_service_status(self, service_name, status):
        """Update the health status of a registered service"""
        if service_name in self.service_status:
            self.service_status[service_name] = status
            
            if status != 'healthy':
                logger.warning(f"Service {service_name} status: {status}")
            
            return True
        else:
            logger.warning(f"Attempted to update unknown service: {service_name}")
            return False

    async def get_performance_history(self, duration_minutes=60):
        """Get historical performance metrics for a time period"""
        if not self.metrics_history:
            return {}
            
        cutoff_time = datetime.now() - pd.Timedelta(minutes=duration_minutes)
        filtered_metrics = [
            m for m in self.metrics_history 
            if datetime.strptime(m['timestamp'], '%Y-%m-%d %H:%M:%S') >= cutoff_time
        ]
        
        if not filtered_metrics:
            return {}
        
        # Calculate averages and peaks
        metrics = {}
        for key in ['cpu_percent', 'memory_percent', 'disk_percent']:
            values = [m[key] for m in filtered_metrics if key in m]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)
                metrics[f"max_{key}"] = max(values)
        
        return metrics

    async def _cleanup(self):
        """Clean up resources used by the system monitor"""
        # Clear any stored metrics data
        self.metrics_history = []
        self.price_history = {}
        
        # Log cleanup
        logger.info("System monitor cleaned up")