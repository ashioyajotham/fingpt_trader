from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAlert:
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    portfolio_impact: Optional[float] = None

class RiskAlertGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.thresholds = config.get('alert_thresholds', {
            'volatility': 0.25,
            'drawdown': 0.15,
            'var_95': 0.10,
            'concentration': 0.40,
            'correlation': 0.80
        })
        
    def generate_alerts(self, risk_metrics: Dict) -> List[RiskAlert]:
        """Generate risk alerts based on current metrics"""
        alerts = []
        
        # Volatility alerts
        if risk_metrics.get('volatility'):
            alerts.extend(self._check_volatility(risk_metrics['volatility']))
            
        # Drawdown alerts
        if risk_metrics.get('max_drawdown'):
            alerts.extend(self._check_drawdown(risk_metrics['max_drawdown']))
            
        # Position concentration alerts
        if risk_metrics.get('concentration'):
            alerts.extend(self._check_concentration(risk_metrics['concentration']))
            
        # Store alerts
        self.alert_history.extend(alerts)
        return alerts
        
    def _check_volatility(self, volatility: float) -> List[RiskAlert]:
        """Check for volatility-related alerts"""
        alerts = []
        threshold = self.thresholds['volatility']
        
        if volatility > threshold * 1.5:
            alerts.append(RiskAlert(
                severity=AlertSeverity.HIGH,
                message=f"Critical volatility level: {volatility:.2%}",
                timestamp=datetime.now(),
                metric_name="volatility",
                current_value=volatility,
                threshold_value=threshold,
                portfolio_impact=-(volatility - threshold)
            ))
        elif volatility > threshold:
            alerts.append(RiskAlert(
                severity=AlertSeverity.MEDIUM,
                message=f"Elevated volatility level: {volatility:.2%}",
                timestamp=datetime.now(),
                metric_name="volatility",
                current_value=volatility,
                threshold_value=threshold
            ))
            
        return alerts
        
    def _check_drawdown(self, drawdown: float) -> List[RiskAlert]:
        """Check for drawdown-related alerts"""
        alerts = []
        threshold = self.thresholds['drawdown']
        
        if drawdown < -threshold:
            severity = AlertSeverity.CRITICAL if drawdown < -threshold * 1.5 else AlertSeverity.HIGH
            alerts.append(RiskAlert(
                severity=severity,
                message=f"Significant drawdown: {drawdown:.2%}",
                timestamp=datetime.now(),
                metric_name="drawdown",
                current_value=drawdown,
                threshold_value=-threshold,
                portfolio_impact=drawdown
            ))
            
        return alerts
        
    def _check_concentration(self, concentration: float) -> List[RiskAlert]:
        """Check for position concentration alerts"""
        alerts = []
        threshold = self.thresholds['concentration']
        
        if concentration > threshold:
            alerts.append(RiskAlert(
                severity=AlertSeverity.MEDIUM,
                message=f"High position concentration: {concentration:.2%}",
                timestamp=datetime.now(),
                metric_name="concentration",
                current_value=concentration,
                threshold_value=threshold
            ))
            
        return alerts
        
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get currently active alerts"""
        recent_alerts = []
        current_time = datetime.now()
        
        for alert in reversed(self.alert_history[-100:]):
            time_diff = (current_time - alert.timestamp).total_seconds()
            if time_diff < 3600:  # Active within last hour
                recent_alerts.append(alert)
                
        return recent_alerts
