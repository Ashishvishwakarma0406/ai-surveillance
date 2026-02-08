"""
WebSocket Service

Manages real-time WebSocket connections for alerts and streaming.
"""

import json
import asyncio
from typing import Set, Dict, Any
from datetime import datetime

from fastapi import WebSocket


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            "alerts": set(),
            "stream": set(),
            "stats": set()
        }
    
    async def connect(self, websocket: WebSocket, channel: str = "alerts"):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if channel in self.subscriptions:
            self.subscriptions[channel].add(websocket)
        
        print(f"📡 WebSocket connected. Channel: {channel}. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        
        for channel in self.subscriptions.values():
            channel.discard(websocket)
        
        print(f"📡 WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict, channel: str = None):
        """Broadcast message to connected clients."""
        targets = self.active_connections
        if channel and channel in self.subscriptions:
            targets = self.subscriptions[channel]
        
        disconnected = set()
        
        for connection in targets:
            try:
                await connection.send_json({
                    **message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                disconnected.add(connection)
        
        # Clean up
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client."""
        try:
            await websocket.send_json({
                **message,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            self.disconnect(websocket)
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "by_channel": {
                channel: len(clients) 
                for channel, clients in self.subscriptions.items()
            }
        }


# Singleton instance
ws_manager = WebSocketManager()
