"""Tests for app/connection_manager.py - WebSocket connection management."""

import asyncio
import time
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from app.connection_manager import ConnectionManager, ServerFullException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_websocket():
    """Create a mock WebSocket object with async methods."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock(return_value={"type": "test"})
    ws.receive_bytes = AsyncMock(return_value=b"test_data")
    return ws


# ---------------------------------------------------------------------------
# ConnectionManager tests
# ---------------------------------------------------------------------------

class TestConnectionManager:
    @pytest.fixture
    def manager(self):
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager):
        """connect() accepts a WebSocket and stores it."""
        ws = make_mock_websocket()
        uid = uuid4()

        await manager.connect(uid, ws)

        ws.accept.assert_awaited_once()
        assert manager.check_user(uid) is True
        assert manager.get_user_count() == 1

    @pytest.mark.asyncio
    async def test_connect_rejects_when_full(self, manager):
        """connect() raises ServerFullException when at capacity."""
        # Fill up with 2 connections
        for _ in range(2):
            ws = make_mock_websocket()
            await manager.connect(uuid4(), ws, max_connections=2)

        # Third should fail
        ws = make_mock_websocket()
        with pytest.raises(ServerFullException):
            await manager.connect(uuid4(), ws, max_connections=2)

        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_removes_user(self, manager):
        """disconnect() removes user from active connections and queues."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        await manager.disconnect(uid)

        assert manager.check_user(uid) is False
        assert manager.get_user_count() == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_user(self, manager):
        """disconnect() is a no-op for unknown users."""
        uid = uuid4()
        await manager.disconnect(uid)  # Should not raise

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(self, manager):
        """disconnect() handles errors from websocket.close()."""
        ws = make_mock_websocket()
        ws.close.side_effect = RuntimeError("connection lost")
        uid = uuid4()
        await manager.connect(uid, ws)

        await manager.disconnect(uid)  # Should not raise
        assert manager.check_user(uid) is False

    @pytest.mark.asyncio
    async def test_send_json(self, manager):
        """send_json() sends data to connected user."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        await manager.send_json(uid, {"msg": "hello"})
        ws.send_json.assert_awaited_once_with({"msg": "hello"})

    @pytest.mark.asyncio
    async def test_send_json_disconnects_on_error(self, manager):
        """send_json() disconnects user if send fails."""
        ws = make_mock_websocket()
        ws.send_json.side_effect = RuntimeError("broken pipe")
        uid = uuid4()
        await manager.connect(uid, ws)

        await manager.send_json(uid, {"msg": "hello"})
        assert manager.check_user(uid) is False

    @pytest.mark.asyncio
    async def test_send_json_nonexistent_user(self, manager):
        """send_json() is a no-op for unknown users."""
        await manager.send_json(uuid4(), {"msg": "hello"})  # Should not raise

    @pytest.mark.asyncio
    async def test_receive_json(self, manager):
        """receive_json() receives data from connected user."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        data = await manager.receive_json(uid)
        assert data == {"type": "test"}

    @pytest.mark.asyncio
    async def test_receive_json_disconnected_user_raises(self, manager):
        """receive_json() raises WebSocketDisconnect for unknown users."""
        from fastapi import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            await manager.receive_json(uuid4())

    @pytest.mark.asyncio
    async def test_receive_bytes(self, manager):
        """receive_bytes() receives binary data from connected user."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        data = await manager.receive_bytes(uid)
        assert data == b"test_data"

    @pytest.mark.asyncio
    async def test_receive_bytes_disconnected_user_raises(self, manager):
        """receive_bytes() raises WebSocketDisconnect for unknown users."""
        from fastapi import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            await manager.receive_bytes(uuid4())

    @pytest.mark.asyncio
    async def test_update_and_get_data(self, manager):
        """update_data() and get_latest_data() work together."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        ns = SimpleNamespace(prompt="hello", strength=0.5)
        await manager.update_data(uid, ns)

        latest = await manager.get_latest_data(uid)
        assert latest.prompt == "hello"
        assert latest.strength == 0.5

    @pytest.mark.asyncio
    async def test_get_latest_data_default(self, manager):
        """get_latest_data() returns empty namespace for unknown user."""
        data = await manager.get_latest_data(uuid4())
        assert isinstance(data, SimpleNamespace)

    @pytest.mark.asyncio
    async def test_check_user(self, manager):
        """check_user() returns correct status."""
        uid = uuid4()
        assert manager.check_user(uid) is False

        ws = make_mock_websocket()
        await manager.connect(uid, ws)
        assert manager.check_user(uid) is True

    @pytest.mark.asyncio
    async def test_multiple_connections(self, manager):
        """Manager handles multiple concurrent connections."""
        users = []
        for _ in range(3):
            ws = make_mock_websocket()
            uid = uuid4()
            await manager.connect(uid, ws)
            users.append(uid)

        assert manager.get_user_count() == 3

        await manager.disconnect(users[1])
        assert manager.get_user_count() == 2
        assert manager.check_user(users[0]) is True
        assert manager.check_user(users[1]) is False
        assert manager.check_user(users[2]) is True


class TestIdleTimeout:
    """Tests for idle connection cleanup."""

    @pytest.fixture
    def manager(self):
        return ConnectionManager(idle_timeout=1.0)

    @pytest.mark.asyncio
    async def test_cleanup_disconnects_idle_users(self, manager):
        """cleanup_idle_connections removes users past the timeout."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        # Simulate time passing beyond the timeout
        manager._last_activity[uid] = time.monotonic() - 2.0

        await manager.cleanup_idle_connections()
        assert manager.check_user(uid) is False
        assert manager.get_user_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active_users(self, manager):
        """cleanup_idle_connections keeps recently active users."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        # Activity is fresh (just connected)
        await manager.cleanup_idle_connections()
        assert manager.check_user(uid) is True

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_disabled(self):
        """cleanup_idle_connections is a no-op when timeout is 0."""
        manager = ConnectionManager(idle_timeout=0)
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        # Force old activity
        manager._last_activity[uid] = time.monotonic() - 99999

        await manager.cleanup_idle_connections()
        assert manager.check_user(uid) is True  # Still connected

    @pytest.mark.asyncio
    async def test_activity_updated_on_send(self, manager):
        """send_json updates last activity timestamp."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        old_time = manager._last_activity[uid]
        await asyncio.sleep(0.01)
        await manager.send_json(uid, {"msg": "ping"})

        assert manager._last_activity[uid] >= old_time

    @pytest.mark.asyncio
    async def test_activity_updated_on_receive(self, manager):
        """receive_json updates last activity timestamp."""
        ws = make_mock_websocket()
        uid = uuid4()
        await manager.connect(uid, ws)

        old_time = manager._last_activity[uid]
        await asyncio.sleep(0.01)
        await manager.receive_json(uid)

        assert manager._last_activity[uid] >= old_time
