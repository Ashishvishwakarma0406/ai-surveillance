"""
FastAPI Main Application

Entry point for the AI Surveillance API.
Run with: uvicorn backend.app.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.api.routes import health, cameras, alerts, incidents
from backend.app.services.websocket_service import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("🚀 Starting AI Surveillance API...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    # Ensure directories exist
    os.makedirs(PROJECT_ROOT / "uploads", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "output", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "storage" / "frames", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "storage" / "clips", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "storage" / "logs", exist_ok=True)
    
    yield
    
    print("👋 Shutting down AI Surveillance API...")


app = FastAPI(
    title="AI Surveillance API",
    description="Real-time video surveillance with AI-powered detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(health.router, tags=["Health"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(incidents.router, prefix="/api/incidents", tags=["Incidents"])


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time alerts and updates."""
    await ws_manager.connect(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to AI Surveillance"
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Stream endpoint
@app.get("/api/stream/video_feed")
async def video_feed(camera_id: str = None, source: str = "webcam"):
    """MJPEG video stream with detections."""
    from backend.app.services.stream_service import stream_manager
    
    return StreamingResponse(
        stream_manager.generate_frames(source, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Static files
try:
    app.mount("/uploads", StaticFiles(directory=str(PROJECT_ROOT / "uploads")), name="uploads")
    app.mount("/output", StaticFiles(directory=str(PROJECT_ROOT / "output")), name="output")
    app.mount("/storage", StaticFiles(directory=str(PROJECT_ROOT / "storage")), name="storage")
except Exception:
    pass  # Directories may not exist yet


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Surveillance API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "frontend": "http://localhost:3000"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
