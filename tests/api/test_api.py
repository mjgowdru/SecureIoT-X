"""
SecureIoT-X — API Tests.

Tests: health, auth, devices, predictions, blockchain.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app import app


@pytest.fixture
def client():
    """Create a test client with fresh Motor client for each loop."""
    import backend.core.mongodb as mongo_mod
    mongo_mod._async_client = None
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        resp = await c.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "services" in data
        assert data["services"]["backend"] == "ONLINE"


@pytest.mark.asyncio
async def test_login_invalid(client):
    async with client as c:
        resp = await c.post("/api/auth/login", json={"username": "nonexistent", "password": "wrong"})
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_devices(client):
    async with client as c:
        resp = await c.get("/api/devices")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_blockchain_status(client):
    async with client as c:
        resp = await c.get("/api/blockchain/status")
        assert resp.status_code == 200
        data = resp.json()
        # Primary blockchain is Ganache
        assert data.get("blockchain") == "ganache"
        assert "ganache" in data


@pytest.mark.asyncio
async def test_list_scenarios(client):
    async with client as c:
        resp = await c.get("/api/simulate/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "scenario_1" in data


@pytest.mark.asyncio
async def test_predict_endpoint(client):
    async with client as c:
        resp = await c.post("/api/predict", json={
            "device_id": "ESP32_TEST",
            "features": {"packet_count": 100, "flow_duration": 1.5},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data


@pytest.mark.asyncio
async def test_simulate_event(client):
    async with client as c:
        resp = await c.post("/api/simulate/event?event_type=normal")
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id" in data
        assert "device_id" in data


@pytest.mark.asyncio
async def test_analytics_overview(client):
    async with client as c:
        resp = await c.get("/api/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_devices" in data
