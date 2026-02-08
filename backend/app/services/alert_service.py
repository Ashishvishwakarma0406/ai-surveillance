"""
Alert Service

Business logic for alert management.
"""

from datetime import datetime
from typing import List, Optional, Dict
from collections import deque

from backend.app.schemas.alert import Alert, AlertCreate, AlertStats, AlertSeverity


class AlertService:
    """
    Alert management service.
    
    In Phase 2, this will use PostgreSQL for persistence.
    """
    
    def __init__(self):
        self._alerts: deque = deque(maxlen=1000)
        self._counter = 0
    
    async def get_alerts(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        acknowledged: Optional[bool] = None
    ) -> List[dict]:
        """Get alerts with filtering."""
        alerts = list(self._alerts)
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if alert_type:
            alerts = [a for a in alerts if a["alert_type"] == alert_type]
        if acknowledged is not None:
            alerts = [a for a in alerts if a["acknowledged"] == acknowledged]
        
        # Most recent first
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    async def create_alert(self, alert: AlertCreate) -> dict:
        """Create a new alert."""
        self._counter += 1
        
        new_alert = {
            "id": self._counter,
            "alert_type": alert.alert_type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "confidence": alert.confidence,
            "camera_id": alert.camera_id,
            "frame_id": alert.frame_id,
            "bbox": alert.bbox,
            "frame_path": alert.frame_path,
            "clip_path": alert.clip_path,
            "metadata": alert.metadata,
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None,
            "timestamp": datetime.now().isoformat()
        }
        
        self._alerts.append(new_alert)
        
        # Broadcast via WebSocket
        await self._broadcast_alert(new_alert)
        
        return new_alert
    
    async def _broadcast_alert(self, alert: dict):
        """Broadcast alert to WebSocket clients."""
        try:
            from backend.app.services.websocket_service import ws_manager
            await ws_manager.broadcast({
                "type": "alert",
                "data": alert
            })
        except Exception:
            pass  # WebSocket not available
    
    async def acknowledge(self, alert_id: int, user_id: str = "admin") -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = user_id
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        return False
    
    async def get_stats(self) -> AlertStats:
        """Get alert statistics."""
        alerts = list(self._alerts)
        
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        unacknowledged = 0
        
        for alert in alerts:
            t = alert["alert_type"]
            by_type[t] = by_type.get(t, 0) + 1
            
            s = alert["severity"]
            by_severity[s] = by_severity.get(s, 0) + 1
            
            if not alert["acknowledged"]:
                unacknowledged += 1
        
        return AlertStats(
            total=len(alerts),
            by_type=by_type,
            by_severity=by_severity,
            recent_24h=len(alerts),
            unacknowledged=unacknowledged
        )
    
    async def clear_all(self):
        """Clear all alerts."""
        self._alerts.clear()


# Singleton instance
alert_service = AlertService()
