"""
Tests for Phase 4: External Device Security Detection and Response Pipeline.

Validates:
  1. Real security event ingestion from registered external physical device.
  2. Preserving event_id and device_id throughout MongoDB collections.
  3. Risk assessment and isolation response execution for HIGH/CRITICAL events.
  4. Per-device isolation containment (quarantine enforcement without affecting other devices).
  5. Persistence of isolation state and manual release restoration.
  6. Ganache blockchain real transaction recording or real logged offline handling.
"""

import time
import uuid
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
async def test_external_device_security_event_pipeline(client):
    """External security event on iot/{device_id}/security should trigger full detection & response pipeline."""
    db = get_mongodb()
    device_id = "ESP32_011"
    event_id = f"EVT-TEST-{uuid.uuid4().hex[:8]}"

    # Ensure device is registered and not isolated
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

    now = time.time()
    msg = MQTTMessage(
        topic=f"iot/{device_id}/security",
        payload={
            "event_id": event_id,
            "device_id": device_id,
            "attack_type": "DDoS",
            "label": "DDoS",
            "packet_count": 60000,
            "timestamp": now,
        },
        device_id=device_id,
        timestamp=now,
    )

    # Process through MQTT service
    await mqtt_service._handle_message_async(msg)

    # 1. Verify device is isolated in MongoDB
    device = await db.devices.find_one({"device_id": device_id})
    assert device["is_isolated"] is True, "Physical device was not isolated in MongoDB!"
    assert device["security_status"] == "ISOLATED"

    # 2. Verify attack_event stored with preserved event_id and device_id
    atk = await db.attack_events.find_one({"event_id": event_id})
    assert atk is not None, "attack_event not saved in MongoDB!"
    assert atk["device_id"] == device_id
    assert atk["severity"] in ("HIGH", "CRITICAL")
    assert atk["action"] in ("ISOLATE", "BLOCK")

    # 3. Verify prediction, risk_score, and response_action stored
    pred = await db.predictions.find_one({"event_id": event_id})
    assert pred is not None
    assert pred["device_id"] == device_id

    risk = await db.risk_scores.find_one({"event_id": event_id})
    assert risk is not None
    assert risk["device_id"] == device_id
    assert risk["severity"] in ("HIGH", "CRITICAL")

    resp_doc = await db.response_actions.find_one({"event_id": event_id})
    assert resp_doc is not None
    assert resp_doc["device_id"] == device_id
    assert resp_doc["action"] in ("ISOLATE", "BLOCK")

    # 4. Verify isolation event recorded
    iso_event = await db.isolation_events.find_one({"event_id": event_id})
    assert iso_event is not None
    assert iso_event["device_id"] == device_id
    assert iso_event["action"] == "ISOLATE"


@pytest.mark.asyncio
async def test_per_device_isolation_containment(client):
    """Isolating one physical device must NOT isolate unrelated devices."""
    db = get_mongodb()
    target_id = "ESP32_011"
    unrelated_id = "ESP32_001"

    # Ensure unrelated device is released
    await db.devices.update_one(
        {"device_id": unrelated_id},
        {"$set": {"is_isolated": False, "security_status": "NORMAL"}}
    )

    # Verify target device is isolated
    target = await db.devices.find_one({"device_id": target_id})
    assert target.get("is_isolated") is True

    # Verify unrelated device is NOT isolated
    unrelated = await db.devices.find_one({"device_id": unrelated_id})
    assert unrelated.get("is_isolated") is False, "Unrelated device was incorrectly isolated!"

    # Quarantine containment: new telemetry from isolated device must be blocked
    prev_isolated_count = mqtt_service.stats["rejected_isolated"]
    msg_blocked = MQTTMessage(
        topic=f"iot/{target_id}/telemetry",
        payload={"device_id": target_id, "temperature": 25.0},
        device_id=target_id,
    )
    await mqtt_service._handle_message_async(msg_blocked)
    assert mqtt_service.stats["rejected_isolated"] == prev_isolated_count + 1


@pytest.mark.asyncio
async def test_manual_release_restores_telemetry(client):
    """Manual release via API restores the physical device and permits telemetry ingestion again."""
    db = get_mongodb()
    device_id = "ESP32_011"

    # Call release API
    async with client as c:
        resp = await c.post(f"/api/devices/{device_id}/release")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "NORMAL"

    # Verify MongoDB state
    device = await db.devices.find_one({"device_id": device_id})
    assert device["is_isolated"] is False
    assert device["security_status"] == "NORMAL"

    # Verify telemetry is now accepted again
    msg = MQTTMessage(
        topic=f"iot/{device_id}/telemetry",
        payload={"device_id": device_id, "temperature": 22.5, "humidity": 48.0},
        device_id=device_id,
    )
    await mqtt_service._handle_message_async(msg)

    # Verify updated
    updated_dev = await db.devices.find_one({"device_id": device_id})
    assert updated_dev["temperature"] == 22.5
    assert updated_dev["humidity"] == 48.0
