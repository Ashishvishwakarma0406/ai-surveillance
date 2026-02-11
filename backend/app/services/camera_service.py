"""
Camera Service

Business logic for camera management and video processing.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from backend.app.schemas.camera import CameraCreate, CameraStatus, CameraType
from backend.app.schemas.alert import AlertCreate, AlertType, AlertSeverity
from backend.app.services.alert_service import alert_service


class CameraService:
    """
    Camera management service.
    
    Handles camera CRUD and video processing orchestration.
    In Phase 2, this will integrate with PostgreSQL.
    """
    
    def __init__(self):
        # In-memory storage (PostgreSQL in Phase 2)
        self._cameras: Dict[str, dict] = {}
        self._jobs: Dict[str, dict] = {}
        self._active_streams: Dict[str, Any] = {}
    
    async def list_cameras(self) -> List[dict]:
        """List all cameras."""
        return list(self._cameras.values())
    
    async def get_camera(self, camera_id: str) -> Optional[dict]:
        """Get camera by ID."""
        return self._cameras.get(camera_id)
    
    async def add_camera(self, camera: CameraCreate) -> dict:
        """Add a new camera."""
        camera_id = str(uuid.uuid4())
        now = datetime.now()
        
        camera_data = {
            "id": camera_id,
            "name": camera.name,
            "type": camera.type.value,
            "url": camera.url,
            "device_index": camera.device_index,
            "location": camera.location,
            "status": CameraStatus.OFFLINE.value,
            "enabled": camera.enabled,
            "fps": None,
            "resolution": None,
            "last_frame_at": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        self._cameras[camera_id] = camera_data
        return camera_data
    
    async def delete_camera(self, camera_id: str) -> dict:
        """Delete a camera."""
        if camera_id in self._cameras:
            # Stop stream if active
            await self.stop_stream(camera_id)
            del self._cameras[camera_id]
            return {"message": "Camera deleted"}
        return {"message": "Camera not found"}
    
    async def start_stream(self, camera_id: str) -> dict:
        """Start camera stream processing."""
        camera = self._cameras.get(camera_id)
        if not camera:
            return {"error": "Camera not found"}
        
        if camera_id in self._active_streams:
            return {"message": "Stream already active"}
        
        camera["status"] = CameraStatus.STREAMING.value
        camera["updated_at"] = datetime.now().isoformat()
        
        # Stream will be managed by stream_manager
        self._active_streams[camera_id] = {
            "started_at": datetime.now().isoformat(),
            "frames_processed": 0
        }
        
        return {
            "message": "Stream started",
            "camera_id": camera_id,
            "stream_url": f"/api/stream/video_feed?camera_id={camera_id}"
        }
    
    async def stop_stream(self, camera_id: str) -> dict:
        """Stop camera stream processing."""
        camera = self._cameras.get(camera_id)
        if camera:
            camera["status"] = CameraStatus.ONLINE.value
            camera["updated_at"] = datetime.now().isoformat()
        
        if camera_id in self._active_streams:
            del self._active_streams[camera_id]
        
        return {"message": "Stream stopped"}
    
    async def process_video(
        self, 
        job_id: str, 
        filepath: str, 
        filename: str
    ):
        """Process an uploaded video file."""
        self._jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "filepath": filepath,
            "status": "processing",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "results": None,
            "error": None
        }
        
        try:
            # Import here to avoid circular imports
            from backend.app.ai.pipelines.video_pipeline import VideoPipeline
            
            pipeline = VideoPipeline()
            
            async for progress, results in pipeline.process(filepath):
                self._jobs[job_id]["progress"] = progress
                if results:
                    self._jobs[job_id]["results"] = results
            
            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["progress"] = 100.0
            self._jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # Save alerts from video processing to alert service
            if self._jobs[job_id].get("results") and self._jobs[job_id]["results"].get("alerts"):
                for alert_data in self._jobs[job_id]["results"]["alerts"]:
                    try:
                        # Map video pipeline alert type to AlertType enum
                        alert_type_str = alert_data.get("type", "anomaly")
                        alert_type = AlertType.WEAPON if alert_type_str == "weapon" else \
                                     AlertType.VIOLENCE if alert_type_str == "violence" else \
                                     AlertType.ANOMALY
                        
                        alert_create = AlertCreate(
                            alert_type=alert_type,
                            severity=AlertSeverity.CRITICAL if alert_data.get("severity") == "critical" else AlertSeverity.WARNING,
                            message=alert_data.get("message", "Alert detected"),
                            confidence=alert_data.get("confidence", 0.0),
                            frame_id=alert_data.get("frame_id"),
                            metadata={
                                "timestamp": alert_data.get("timestamp"),
                                "job_id": job_id,
                                "source": "video_upload"
                            }
                        )
                        await alert_service.create_alert(alert_create)
                    except Exception as e:
                        print(f"⚠️ Failed to save alert: {e}")
            
        except Exception as e:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = str(e)
    
    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get video processing job status."""
        job = self._jobs.get(job_id)
        if job:
            # Add video URL for frontend playback
            results = job.get("results")
            if results and results.get("output_video"):
                # Use the annotated output video
                output_path = results["output_video"]
                output_filename = Path(output_path).name
                job["video_url"] = f"/output/{output_filename}"
            else:
                # Fallback to original upload
                filepath = job.get("filepath", "")
                if filepath:
                    filename = Path(filepath).name
                    job["video_url"] = f"/uploads/{filename}"
        return job
