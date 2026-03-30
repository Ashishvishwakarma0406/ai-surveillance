"""
Alert Schemas

Pydantic models for alert-related API requests/responses.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    WEAPON = "weapon"
    VIOLENCE = "violence"
    CROWD = "crowd"
    TRASH = "trash"
    INTRUSION = "intrusion"
    ANOMALY = "anomaly"
    ACCIDENT = "accident"
    VEHICLE = "vehicle"


class AlertCategory(str, Enum):
    """High-level grouping of alerts into two features."""
    VIOLENCE = "violence"   # weapons, fighting, crowds, danger
    TRAFFIC = "traffic"     # vehicle collisions, crashes


# Map AlertType -> AlertCategory for auto-classification
ALERT_TYPE_TO_CATEGORY = {
    AlertType.WEAPON: AlertCategory.VIOLENCE,
    AlertType.VIOLENCE: AlertCategory.VIOLENCE,
    AlertType.CROWD: AlertCategory.VIOLENCE,
    AlertType.INTRUSION: AlertCategory.VIOLENCE,
    AlertType.ANOMALY: AlertCategory.VIOLENCE,
    AlertType.TRASH: AlertCategory.VIOLENCE,
    AlertType.ACCIDENT: AlertCategory.TRAFFIC,
    AlertType.VEHICLE: AlertCategory.TRAFFIC,
}


class AlertCreate(BaseModel):
    """Create alert request."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    confidence: float = Field(..., ge=0, le=1)
    category: Optional[AlertCategory] = None  # Auto-derived if not set
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    bbox: Optional[Any] = None  # [x1, y1, x2, y2] or {x1, y1, x2, y2}
    frame_path: Optional[str] = None
    clip_path: Optional[str] = None
    metadata: Optional[Dict] = None


class Alert(BaseModel):
    """Alert response."""
    id: int
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    confidence: float
    category: Optional[str] = None
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    bbox: Optional[Any] = None
    frame_path: Optional[str] = None
    clip_path: Optional[str] = None
    metadata: Optional[Dict] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class AlertStats(BaseModel):
    """Alert statistics."""
    total: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_category: Dict[str, int] = {}
    recent_24h: int
    unacknowledged: int
