"""
Camera & Video Routes

Handles camera management and video uploads.
"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.services.camera_service import CameraService
from backend.app.schemas.camera import CameraCreate, CameraResponse, CameraStatus

router = APIRouter()
camera_service = CameraService()


class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str


@router.get("/")
async def list_cameras():
    """List all configured cameras."""
    return await camera_service.list_cameras()


@router.post("/")
async def add_camera(camera: CameraCreate):
    """Add a new camera source."""
    return await camera_service.add_camera(camera)


@router.get("/{camera_id}")
async def get_camera(camera_id: str):
    """Get camera details."""
    camera = await camera_service.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera."""
    return await camera_service.delete_camera(camera_id)


@router.post("/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start camera stream processing."""
    return await camera_service.start_stream(camera_id)


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop camera stream processing."""
    return await camera_service.stop_stream(camera_id)


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a video file for processing."""
    print(f"📥 Received upload request: {file.filename}")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        print(f"❌ Invalid extension: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    job_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / f"{job_id}{ext}"
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    try:
        content = await file.read()
        size = len(content)
        print(f"📄 File size: {size / (1024*1024):.2f} MB")
        
        if size > settings.MAX_UPLOAD_SIZE:
            print(f"❌ File too large: {size}")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
            )
        
        with open(upload_path, "wb") as f:
            f.write(content)
        print(f"✅ File saved to {upload_path}")
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Queue for background processing
    background_tasks.add_task(
        camera_service.process_video,
        job_id,
        str(upload_path),
        file.filename
    )
    
    return VideoUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="queued",
        message="Video uploaded. Processing started."
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get video processing job status."""
    return await camera_service.get_job_status(job_id)
