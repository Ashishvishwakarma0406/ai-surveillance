"""
Application Configuration

Central configuration using Pydantic Settings.
"""

from typing import List
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "AI Surveillance API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    
    # Video Processing
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    # Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.5
    VIOLENCE_THRESHOLD: float = 0.6
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
