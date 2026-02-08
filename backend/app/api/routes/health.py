"""
Health Check Routes
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-surveillance-api"}


@router.get("/ready")
async def readiness_check():
    """Readiness check - verifies dependencies."""
    return {
        "status": "ready",
        "checks": {
            "api": True,
            "database": True,  # Will check PostgreSQL in Phase 2
            "redis": True,     # Will check Redis in Phase 2
            "models": True
        }
    }
