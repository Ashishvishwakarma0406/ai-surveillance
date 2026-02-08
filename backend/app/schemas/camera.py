"""
Camera Schemas

Pydantic models for camera-related API requests/responses.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class CameraType(str, Enum):
    WEBCAM = "webcam"
    RTSP = "rtsp"
    FILE = "file"


class CameraStatus(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    STREAMING = "streaming"
    ERROR = "error"


class CameraCreate(BaseModel):
    """Create camera request."""
    name: str = Field(..., min_length=1, max_length=100)
    type: CameraType
    url: Optional[str] = None  # For RTSP
    device_index: Optional[int] = 0  # For webcam
    location: Optional[str] = None
    enabled: bool = True


class CameraUpdate(BaseModel):
    """Update camera request."""
    name: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None
    enabled: Optional[bool] = None


class CameraResponse(BaseModel):
    """Camera response."""
    id: str
    name: str
    type: CameraType
    url: Optional[str] = None
    device_index: Optional[int] = None
    location: Optional[str] = None
    status: CameraStatus
    enabled: bool
    fps: Optional[float] = None
    resolution: Optional[str] = None
    last_frame_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProcessingJob(BaseModel):
    """Video processing job."""
    job_id: str
    filename: str
    status: str  # pending, processing, completed, failed
    progress: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[dict] = None
    error: Optional[str] = None
