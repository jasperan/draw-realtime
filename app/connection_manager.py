"""WebSocket connection management."""

import asyncio
from typing import Dict, Any, Optional
from uuid import UUID
from fastapi import WebSocket, WebSocketDisconnect
from types import SimpleNamespace


class ServerFullException(Exception):
    """Raised when server has reached maximum connections."""
    pass


class ConnectionManager:
    """Manages WebSocket connections and their data queues."""

    def __init__(self):
        self.active_connections: Dict[UUID, WebSocket] = {}
        self.data_queues: Dict[UUID, SimpleNamespace] = {}

    async def connect(self, user_id: UUID, websocket: WebSocket, max_connections: int = 4):
        """Accept a new WebSocket connection."""
        if len(self.active_connections) >= max_connections:
            await websocket.close(code=1013, reason="Server is full")
            raise ServerFullException("Maximum connections reached")

        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.data_queues[user_id] = SimpleNamespace()

    async def disconnect(self, user_id: UUID):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].close()
            except Exception:
                pass
            del self.active_connections[user_id]

        if user_id in self.data_queues:
            del self.data_queues[user_id]

    def check_user(self, user_id: UUID) -> bool:
        """Check if a user is connected."""
        return user_id in self.active_connections

    async def send_json(self, user_id: UUID, data: Dict[str, Any]):
        """Send JSON data to a user."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
            except Exception:
                await self.disconnect(user_id)

    async def receive_json(self, user_id: UUID) -> Dict[str, Any]:
        """Receive JSON data from a user."""
        if user_id not in self.active_connections:
            raise WebSocketDisconnect()
        return await self.active_connections[user_id].receive_json()

    async def receive_bytes(self, user_id: UUID) -> bytes:
        """Receive binary data from a user."""
        if user_id not in self.active_connections:
            raise WebSocketDisconnect()
        return await self.active_connections[user_id].receive_bytes()

    async def update_data(self, user_id: UUID, data: SimpleNamespace):
        """Update the latest data for a user."""
        self.data_queues[user_id] = data

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        """Get the latest data for a user."""
        return self.data_queues.get(user_id, SimpleNamespace())

    def get_user_count(self) -> int:
        """Get the number of connected users."""
        return len(self.active_connections)
