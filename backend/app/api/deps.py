"""
API Dependencies

Common dependencies for route handlers.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Security scheme (will be implemented in Phase 2 with auth)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Get current authenticated user.
    
    For MVP, returns a mock user. Will integrate with auth in Phase 2.
    """
    # MVP: Return mock user
    return {
        "id": "user-1",
        "email": "admin@surveillance.local",
        "role": "admin"
    }


async def require_auth(
    user: dict = Depends(get_current_user)
):
    """Require authentication for protected routes."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


async def require_admin(
    user: dict = Depends(require_auth)
):
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user
