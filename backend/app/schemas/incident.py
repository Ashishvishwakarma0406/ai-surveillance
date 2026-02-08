"""
Incident Schemas

Pydantic models for incident-related API requests/responses.
"""

from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_ALARM = "false_alarm"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentCreate(BaseModel):
    """Create incident request."""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    severity: IncidentSeverity
    camera_id: Optional[str] = None
    alert_ids: List[int] = []
    metadata: Optional[Dict] = None


class IncidentUpdate(BaseModel):
    """Update incident request."""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    notes: Optional[str] = None
    resolved_by: Optional[str] = None


class Incident(BaseModel):
    """Incident response."""
    id: str
    title: str
    description: Optional[str] = None
    status: IncidentStatus
    severity: IncidentSeverity
    camera_id: Optional[str] = None
    alert_ids: List[int] = []
    clip_paths: List[str] = []
    frame_paths: List[str] = []
    notes: Optional[str] = None
    metadata: Optional[Dict] = None
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    class Config:
        from_attributes = True


class TimelineEvent(BaseModel):
    """Incident timeline event."""
    timestamp: datetime
    event_type: str
    description: str
    data: Optional[Dict] = None
