"""
Incident Service

Business logic for incident management.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict

from backend.app.schemas.incident import (
    Incident, IncidentCreate, IncidentUpdate, 
    IncidentStatus, TimelineEvent
)


class IncidentService:
    """
    Incident management service.
    
    Groups related alerts into incidents for investigation.
    """
    
    def __init__(self):
        self._incidents: Dict[str, dict] = {}
        self._timelines: Dict[str, List[dict]] = {}
    
    async def list_incidents(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[dict]:
        """List incidents with filtering."""
        incidents = list(self._incidents.values())
        
        if status:
            incidents = [i for i in incidents if i["status"] == status]
        if severity:
            incidents = [i for i in incidents if i["severity"] == severity]
        
        # Most recent first
        return sorted(
            incidents, 
            key=lambda x: x["created_at"], 
            reverse=True
        )[:limit]
    
    async def get_incident(self, incident_id: str) -> Optional[dict]:
        """Get incident by ID."""
        return self._incidents.get(incident_id)
    
    async def create_incident(self, incident: IncidentCreate) -> dict:
        """Create a new incident."""
        incident_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        new_incident = {
            "id": incident_id,
            "title": incident.title,
            "description": incident.description,
            "status": IncidentStatus.OPEN.value,
            "severity": incident.severity.value,
            "camera_id": incident.camera_id,
            "alert_ids": incident.alert_ids,
            "clip_paths": [],
            "frame_paths": [],
            "notes": None,
            "metadata": incident.metadata,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
            "resolved_by": None
        }
        
        self._incidents[incident_id] = new_incident
        
        # Initialize timeline
        self._timelines[incident_id] = [{
            "timestamp": now,
            "event_type": "created",
            "description": "Incident created",
            "data": None
        }]
        
        return new_incident
    
    async def update_incident(
        self, 
        incident_id: str, 
        update: IncidentUpdate
    ) -> Optional[dict]:
        """Update incident."""
        incident = self._incidents.get(incident_id)
        if not incident:
            return None
        
        now = datetime.now().isoformat()
        
        if update.title:
            incident["title"] = update.title
        if update.description:
            incident["description"] = update.description
        if update.status:
            old_status = incident["status"]
            incident["status"] = update.status.value
            self._add_timeline_event(incident_id, "status_change", 
                f"Status: {old_status} → {update.status.value}")
        if update.severity:
            incident["severity"] = update.severity.value
        if update.notes:
            incident["notes"] = update.notes
        if update.resolved_by:
            incident["resolved_by"] = update.resolved_by
            incident["resolved_at"] = now
        
        incident["updated_at"] = now
        
        return incident
    
    def _add_timeline_event(
        self, 
        incident_id: str, 
        event_type: str, 
        description: str,
        data: Optional[dict] = None
    ):
        """Add event to incident timeline."""
        if incident_id not in self._timelines:
            self._timelines[incident_id] = []
        
        self._timelines[incident_id].append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "data": data
        })
    
    async def get_clips(self, incident_id: str) -> List[str]:
        """Get video clips for incident."""
        incident = self._incidents.get(incident_id)
        return incident.get("clip_paths", []) if incident else []
    
    async def get_timeline(self, incident_id: str) -> List[dict]:
        """Get timeline for incident."""
        return self._timelines.get(incident_id, [])


# Singleton instance
incident_service = IncidentService()
