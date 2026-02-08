"""
Incident Routes

Manages recorded incidents (grouped alerts).
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException

from backend.app.services.incident_service import IncidentService
from backend.app.schemas.incident import Incident, IncidentCreate, IncidentUpdate

router = APIRouter()
incident_service = IncidentService()


@router.get("/", response_model=List[Incident])
async def list_incidents(
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = None,
    severity: Optional[str] = None
):
    """List all incidents."""
    return await incident_service.list_incidents(
        limit=limit,
        status=status,
        severity=severity
    )


@router.get("/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str):
    """Get incident details."""
    incident = await incident_service.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@router.post("/", response_model=Incident)
async def create_incident(incident: IncidentCreate):
    """Create a new incident."""
    return await incident_service.create_incident(incident)


@router.patch("/{incident_id}", response_model=Incident)
async def update_incident(incident_id: str, update: IncidentUpdate):
    """Update incident status/notes."""
    incident = await incident_service.update_incident(incident_id, update)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@router.get("/{incident_id}/clips")
async def get_incident_clips(incident_id: str):
    """Get video clips associated with incident."""
    return await incident_service.get_clips(incident_id)


@router.get("/{incident_id}/timeline")
async def get_incident_timeline(incident_id: str):
    """Get timeline of events for incident."""
    return await incident_service.get_timeline(incident_id)
