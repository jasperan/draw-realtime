"""WebSocket connection management."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from uuid import UUID
from fastapi import WebSocket, WebSocketDisconnect
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class ServerFullException(Exception):
    """Raised when server has reached maximum connections."""
    pass


class ConnectionManager:
    """Manages WebSocket connections and their data queues."""

    def __init__(self, idle_timeout: float = 300.0):
        self.active_connections: Dict[UUID, WebSocket] = {}
        self.data_queues: Dict[UUID, SimpleNamespace] = {}
        self._last_activity: Dict[UUID, float] = {}
        self._idle_timeout = idle_timeout  # seconds, 0 = no timeout

    async def connect(self, user_id: UUID, websocket: WebSocket, max_connections: int = 4):
        """Accept a new WebSocket connection."""
        if len(self.active_connections) >= max_connections:
            await websocket.close(code=1013, reason="Server is full")
            raise ServerFullException("Maximum connections reached")

        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.data_queues[user_id] = SimpleNamespace()
        self._last_activity[user_id] = time.monotonic()

    async def disconnect(self, user_id: UUID):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].close()
            except Exception:
                pass
            del self.active_connections[user_id]

        self.data_queues.pop(user_id, None)
        self._last_activity.pop(user_id, None)

    def _touch(self, user_id: UUID):
        """Update last activity timestamp for a user."""
        if user_id in self._last_activity:
            self._last_activity[user_id] = time.monotonic()

    def check_user(self, user_id: UUID) -> bool:
        """Check if a user is connected."""
        return user_id in self.active_connections

    async def send_json(self, user_id: UUID, data: Dict[str, Any]):
        """Send JSON data to a user."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
                self._touch(user_id)
            except Exception:
                await self.disconnect(user_id)

    async def receive_json(self, user_id: UUID) -> Dict[str, Any]:
        """Receive JSON data from a user."""
        if user_id not in self.active_connections:
            raise WebSocketDisconnect()
        data = await self.active_connections[user_id].receive_json()
        self._touch(user_id)
        return data

    async def receive_bytes(self, user_id: UUID) -> bytes:
        """Receive binary data from a user."""
        if user_id not in self.active_connections:
            raise WebSocketDisconnect()
        data = await self.active_connections[user_id].receive_bytes()
        self._touch(user_id)
        return data

    async def update_data(self, user_id: UUID, data: SimpleNamespace):
        """Update the latest data for a user."""
        self.data_queues[user_id] = data

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        """Get the latest data for a user."""
        return self.data_queues.get(user_id, SimpleNamespace())

    def get_user_count(self) -> int:
        """Get the number of connected users."""
        return len(self.active_connections)

    async def cleanup_idle_connections(self):
        """Disconnect users that have been idle beyond the timeout."""
        if self._idle_timeout <= 0:
            return

        now = time.monotonic()
        idle_users = [
            uid for uid, last in self._last_activity.items()
            if (now - last) > self._idle_timeout
        ]
        for uid in idle_users:
            logger.info(f"Disconnecting idle user {uid}")
            await self.disconnect(uid)
