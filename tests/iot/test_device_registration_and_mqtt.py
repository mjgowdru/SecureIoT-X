"""
Tests for Physical Device Registration and MQTT Ingestion.

Validates Phase 2 implementation:
  1. Registering physical devices with is_simulation=False.
  2. Rejecting duplicate or reserved simulation IDs (ESP32_001 - ESP32_010).
  3. Rejecting unauthorized/unregistered device MQTT telemetry.
  4. Detecting and dropping spoofed device IDs.
  5. Dropping telemetry from quarantined/isolated devices.
  6. Message de-duplication and payload sanitization.
  7. LWT status handling (online/offline state persistence).
"""

import asyncio
import time
import pytest
from httpx import AsyncClient, ASGITransport

from backend.app import app
from backend.core.mongodb import get_mongodb, init_mongodb
from backend.services.mqtt_service import mqtt_service
from iot.mqtt_client import MQTTMessage


@pytest.fixture(autouse=True)
def reset_motor():
    """Ensure Motor client binds to current test loop."""
    import backend.core.mongodb as mongo_mod
    mongo_mod._async_client = None
    yield
    mongo_mod._async_client = None


@pytest.fixture(autouse=True)
def reset_network_enforcer():
    """Reset NetworkEnforcementManager singleton state between tests."""
    from security.network_enforcer import network_enforcer
    network_enforcer._active_blocks.clear()
    network_enforcer._blocked_ips.clear()
    yield
    network_enforcer._active_blocks.clear()
    network_enforcer._blocked_ips.clear()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_register_physical_device_success(client):
    """Registering a new physical device like ESP32_011 should succeed and set is_simulation=False."""
    db = get_mongodb()
    # Clean up test device if present
    await db.devices.delete_one({"device_id": "ESP32_011"})

    async with client as c:
        resp = await c.post("/api/devices", json={
            "device_id": "ESP32_011",
            "name": "Lab Physical DHT22 Sensor",
            "device_type": "ESP32-WROOM-32",
            "ip_address": "10.108.133.55",
            "mac_address": "24:6F:28:AB:CD:EF",
            "location": "Hardware Lab 1",
            "is_simulation": False,
            "telemetry_schema": {"temperature": "float", "humidity": "float", "rssi": "float"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "ESP32_011"
        assert data["is_simulation"] is False

    # Check MongoDB directly
    dev_doc = await db.devices.find_one({"device_id": "ESP32_011"})
    assert dev_doc is not None
    assert dev_doc["is_simulation"] is False
    assert dev_doc["device_type"] == "ESP32-WROOM-32"
    assert dev_doc["mac_address"] == "24:6F:28:AB:CD:EF"
    assert dev_doc["is_isolated"] is False


@pytest.mark.asyncio
async def test_register_reserved_simulation_id_rejected(client):
    """External registration must NOT claim or impersonate reserved simulation IDs (ESP32_001-ESP32_010)."""
    async with client as c:
        resp = await c.post("/api/devices", json={
            "device_id": "ESP32_001",
            "name": "Spoofed Simulation Device",
            "device_type": "ESP32",
            "is_simulation": False,
        })
        assert resp.status_code == 400
        assert "reserved for simulation" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_duplicate_device_rejected(client):
    """Registering an already-registered device must fail with 400."""
    db = get_mongodb()
    await db.devices.delete_one({"device_id": "ESP32_012"})

    async with client as c:
        resp1 = await c.post("/api/devices", json={"device_id": "ESP32_012", "is_simulation": False})
        assert resp1.status_code == 200
        resp2 = await c.post("/api/devices", json={"device_id": "ESP32_012", "is_simulation": False})
        assert resp2.status_code == 400
        assert "already registered" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_mqtt_ingestion_valid_physical_telemetry():
    """Valid telemetry from a registered device should update MongoDB and device state."""
    db = get_mongodb()
    device_id = "ESP32_011"

    # Ensure device exists
    await db.devices.update_one(
        {"device_id": device_id},
        {"$set": {"device_id": device_id, "is_simulation": False, "is_online": True, "is_isolated": False}},
        upsert=True,
    )

    now = time.time()
    msg = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={
            "device_id": device_id,
            "temperature": 27.8,
            "humidity": 55.4,
            "rssi": -65.0,
            "uptime": 120.0,
            "timestamp": now,
        },
        device_id=device_id,
        timestamp=now,
    )

    await mqtt_service._handle_message_async(msg)

    # Verify device record updated in MongoDB
    device = await db.devices.find_one({"device_id": device_id})
    assert device["temperature"] == 27.8
    assert device["humidity"] == 55.4
    assert device["is_online"] is True
    assert device["mqtt_connected"] is True

    # Verify record in telemetry collection
    telem = await db.telemetry.find_one({"device_id": device_id}, sort=[("timestamp", -1)])
    assert telem is not None
    assert telem["temperature"] == 27.8
    assert telem["humidity"] == 55.4


@pytest.mark.asyncio
async def test_mqtt_ingestion_rejects_unregistered_device():
    """Telemetry from an unregistered device ID must be dropped and counted in statistics."""
    db = get_mongodb()
    unreg_id = "ESP32_ROGUE_999"
    await db.devices.delete_one({"device_id": unreg_id})

    prev_rejected = mqtt_service.stats["rejected_unregistered"]
    msg = MQTTMessage(
        topic=f"iot/{unreg_id}/telemetry",
        payload={"device_id": unreg_id, "temperature": 25.0},
        device_id=unreg_id,
    )

    await mqtt_service._handle_message_async(msg)
    assert mqtt_service.stats["rejected_unregistered"] == prev_rejected + 1

    # Verify no telemetry was saved
    telem = await db.telemetry.find_one({"device_id": unreg_id})
    assert telem is None


@pytest.mark.asyncio
async def test_mqtt_ingestion_detects_spoofed_device_id():
    """Mismatched topic and payload device ID must be detected and dropped."""
    prev_spoofed = mqtt_service.stats["rejected_spoofed"]
    msg = MQTTMessage(
        topic="iot/ESP32_011/telemetry",
        payload={"device_id": "ESP32_001", "temperature": 30.0},
        device_id="ESP32_001",
    )

    await mqtt_service._handle_message_async(msg)
    assert mqtt_service.stats["rejected_spoofed"] == prev_spoofed + 1


@pytest.mark.asyncio
async def test_mqtt_ingestion_enforces_isolation_quarantine():
    """An isolated device must have its MQTT traffic blocked at ingestion boundary."""
    db = get_mongodb()
    device_id = "ESP32_011"

    # Set device as isolated
    await db.devices.update_one(
        {"device_id": device_id},
        {"$set": {"is_isolated": True, "security_status": "ISOLATED"}}
    )

    prev_isolated = mqtt_service.stats["rejected_isolated"]
    msg = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={"device_id": device_id, "temperature": 99.0},
        device_id=device_id,
        timestamp=time.time(),
    )

    await mqtt_service._handle_message_async(msg)
    assert mqtt_service.stats["rejected_isolated"] == prev_isolated + 1

    # Clean up: un-isolate device
    await db.devices.update_one(
        {"device_id": device_id},
        {"$set": {"is_isolated": False, "security_status": "NORMAL"}}
    )


@pytest.mark.asyncio
async def test_mqtt_ingestion_lwt_status_offline():
    """LWT offline message should mark device offline in MongoDB."""
    db = get_mongodb()
    device_id = "ESP32_011"

    msg = MQTTMessage(
        topic=f"iot/{device_id}/status",
        payload={"device_id": device_id, "status": "offline"},
        device_id=device_id,
    )

    await mqtt_service._handle_message_async(msg)

    device = await db.devices.find_one({"device_id": device_id})
    assert device["is_online"] is False
    assert device["mqtt_connected"] is False
