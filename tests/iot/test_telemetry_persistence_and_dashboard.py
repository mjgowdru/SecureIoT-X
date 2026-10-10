"""
Tests for Phase 3: Telemetry Persistence and Dashboard Visibility.

Validates:
  1. Telemetry query endpoints (per-device and recent fleet).
  2. Analytics overview physical/simulation device counts.
  3. Telemetry values (temperature, humidity, motion, rssi) persisted accurately.
  4. Non-intrusion sensor telemetry is not falsely treated as an attack.
"""

import time
import pytest
from httpx import AsyncClient, ASGITransport

from backend.app import app
from backend.core.mongodb import get_mongodb
from backend.services.mqtt_service import mqtt_service
from iot.mqtt_client import MQTTMessage


@pytest.fixture(autouse=True)
def reset_motor():
    """Reset Motor client singleton before and after each test."""
    import backend.core.mongodb as mongo_mod
    mongo_mod._async_client = None
    yield
    mongo_mod._async_client = None


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_telemetry_persistence_and_query(client):
    """Publish multiple telemetry points for a physical device and query via API."""
    db = get_mongodb()
    device_id = "ESP32_011"

    # Register/ensure device
    await db.devices.update_one(
        {"device_id": device_id},
        {"$set": {
            "device_id": device_id,
            "name": "Lab Sensor",
            "is_simulation": False,
            "is_online": True,
            "is_isolated": False,
        }},
        upsert=True,
    )

    # Publish two telemetry readings with distinct timestamps
    t1 = time.time() - 10
    t2 = time.time()

    msg1 = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={"device_id": device_id, "temperature": 24.5, "humidity": 50.0, "timestamp": t1},
        device_id=device_id,
        timestamp=t1,
    )
    msg2 = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={"device_id": device_id, "temperature": 25.2, "humidity": 52.1, "timestamp": t2},
        device_id=device_id,
        timestamp=t2,
    )

    await mqtt_service._handle_message_async(msg1)
    await mqtt_service._handle_message_async(msg2)

    # Query device telemetry endpoint
    async with client as c:
        resp = await c.get(f"/api/devices/{device_id}/telemetry?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        # Most recent reading first
        assert data[0]["temperature"] == 25.2
        assert data[0]["humidity"] == 52.1

        # Query recent fleet telemetry endpoint
        resp_recent = await c.get("/api/devices/telemetry/recent?limit=10")
        assert resp_recent.status_code == 200
        recent_data = resp_recent.json()
        assert len(recent_data) >= 2


@pytest.mark.asyncio
async def test_analytics_overview_counts(client):
    """Overview endpoint must report total, physical, and simulation device counts."""
    from backend.core.mongodb import init_mongodb
    await init_mongodb()
    db = get_mongodb()
    # Ensure at least one physical device
    await db.devices.update_one(
        {"device_id": "ESP32_011"},
        {"$set": {"device_id": "ESP32_011", "is_simulation": False}},
        upsert=True,
    )

    async with client as c:
        resp = await c.get("/api/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_devices" in data
        assert "physical_devices" in data
        assert "simulation_devices" in data
        assert data["physical_devices"] >= 1
        assert data["simulation_devices"] >= 10
        assert data["total_devices"] == data["physical_devices"] + data["simulation_devices"]


@pytest.mark.asyncio
async def test_telemetry_does_not_fabricate_attack():
    """Normal sensor telemetry must not be classified as an attack event."""
    db = get_mongodb()
    device_id = "ESP32_011"

    # Reset device to NORMAL state before this test (guard against inter-test contamination)
    await db.devices.update_one(
        {"device_id": device_id},
        {"$set": {
            "device_id": device_id,
            "is_simulation": False,
            "is_isolated": False,
            "security_status": "NORMAL",
            "is_online": True,
        }},
        upsert=True,
    )

    prev_attacks = await db.attack_events.count_documents({"device_id": device_id})

    # Send normal reading
    msg = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={"device_id": device_id, "temperature": 23.0, "humidity": 45.0},
        device_id=device_id,
        timestamp=time.time(),
    )
    await mqtt_service._handle_message_async(msg)

    # Ensure no attack event was created
    new_attacks = await db.attack_events.count_documents({"device_id": device_id})
    assert new_attacks == prev_attacks, "Normal sensor telemetry was falsely classified as an attack!"

    # Device must remain NORMAL
    dev = await db.devices.find_one({"device_id": device_id})
    assert dev["security_status"] == "NORMAL"
    assert dev["is_isolated"] is False
