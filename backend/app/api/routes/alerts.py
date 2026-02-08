"""
Alert Routes

Manages detection alerts.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException

from backend.app.services.alert_service import AlertService
from backend.app.schemas.alert import Alert, AlertCreate, AlertStats

router = APIRouter()
alert_service = AlertService()


@router.get("/", response_model=List[Alert])
async def get_alerts(
    limit: int = Query(50, ge=1, le=100),
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    acknowledged: Optional[bool] = None
):
    """Get alerts with optional filtering."""
    return await alert_service.get_alerts(
        limit=limit,
        severity=severity,
        alert_type=alert_type,
        acknowledged=acknowledged
    )


@router.post("/", response_model=Alert)
async def create_alert(alert: AlertCreate):
    """Create a new alert (used by detection pipeline)."""
    return await alert_service.create_alert(alert)


@router.get("/stats", response_model=AlertStats)
async def get_alert_stats():
    """Get alert statistics."""
    return await alert_service.get_stats()


@router.patch("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    result = await alert_service.acknowledge(alert_id)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}


@router.delete("/clear")
async def clear_alerts():
    """Clear all alerts."""
    await alert_service.clear_all()
    return {"message": "All alerts cleared"}
