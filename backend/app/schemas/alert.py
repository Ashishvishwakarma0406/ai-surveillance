"""
Alert Schemas

Pydantic models for alert-related API requests/responses.
"""

from typing import Optional, Dict
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


class AlertCreate(BaseModel):
    """Create alert request."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    confidence: float = Field(..., ge=0, le=1)
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    bbox: Optional[Dict] = None  # {x1, y1, x2, y2}
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
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    bbox: Optional[Dict] = None
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
    recent_24h: int
    unacknowledged: int
