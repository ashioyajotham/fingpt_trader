import psutil
import torch
import logging
from typing import Dict, List
from datetime import datetime
import asyncio

class SystemMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = []
        self.alert_thresholds = config.get('alert_thresholds', {
            'gpu_memory_percent': 90,
            'cpu_usage_percent': 80,
            'processing_time_seconds': 5
        })
        
    async def monitor_resources(self) -> Dict:
        """Monitor system resources"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory'] = {
                i: torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100
                for i in range(torch.cuda.device_count())
            }
            
        self.metrics_history.append(metrics)
        return metrics
        
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        if metrics['cpu_percent'] > self.alert_thresholds['cpu_usage_percent']:
            alerts.append({
                'level': 'warning',
                'type': 'cpu_usage',
                'message': f"CPU usage at {metrics['cpu_percent']}%"
            })
            
        if 'gpu_memory' in metrics:
            for device, usage in metrics['gpu_memory'].items():
                if usage > self.alert_thresholds['gpu_memory_percent']:
                    alerts.append({
                        'level': 'warning',
                        'type': 'gpu_memory',
                        'message': f"GPU {device} memory at {usage:.1f}%"
                    })
                    
        return alerts