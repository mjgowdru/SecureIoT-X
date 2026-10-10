"""
Simulation E2E Test — Full pipeline: attack → SVM → risk → isolation → MongoDB → Ganache.

Tests the actual FastAPI simulation API endpoints against real MongoDB and Ganache.
Requires both services running.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from backend.app import app


@pytest.fixture
async def client():
    """Create async test client for FastAPI app.

    Resets the Motor MongoDB client singleton so it binds to the current
    event loop (avoids 'Event loop is closed' errors across test classes).
    """
    import backend.core.mongodb as mongo_mod
    mongo_mod._async_client = None  # Reset so Motor binds to *this* loop

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestSimulationPipeline:
    """Test the simulation pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Health endpoint should report MongoDB and Ganache status."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        # Must have explicit MongoDB and Ganache status
        assert "mongodb" in data, "Health missing 'mongodb' field"
        assert "ganache" in data, "Health missing 'ganache' field"
        assert data["mongodb"] in ("CONNECTED", "NOT CONNECTED")
        assert data["ganache"] in ("CONNECTED", "NOT CONNECTED")

    @pytest.mark.asyncio
    async def test_list_simulation_devices(self, client):
        """Should load simulation devices from MongoDB."""
        resp = await client.get("/api/simulate/devices")
        assert resp.status_code == 200
        devices = resp.json()
        assert isinstance(devices, list)
        # Should have seeded devices (10 by default)
        assert len(devices) >= 1, "No simulation devices found — is MongoDB running?"
        # Each device should have required fields
        for d in devices:
            assert "device_id" in d
            assert "is_isolated" in d
            assert "security_status" in d

    @pytest.mark.asyncio
    async def test_attack_simulation_full_pipeline(self, client):
        """
        Full attack pipeline for a single device:
        Attack → SVM → Risk → Response → Isolation → MongoDB → Ganache
        """
        resp = await client.post("/api/simulate/attack", json={
            "device_id": "ESP32_003",
            "attack_type": "DDoS",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Must have real event ID (not hardcoded)
        assert "event_id" in data
        assert len(data["event_id"]) > 0

        # Must have SVM prediction (real model or fallback)
        assert "prediction" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0

        # Must have model info
        assert "model" in data

        # Must have risk assessment
        assert "risk_score" in data
        assert "severity" in data
        assert data["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

        # Must have response action
        assert "action" in data

        # Must have pipeline trace (real backend steps)
        assert "pipeline" in data
        assert isinstance(data["pipeline"], list)
        assert len(data["pipeline"]) >= 3  # At least TRAFFIC, SVM, RISK steps

        # Check pipeline steps exist
        step_names = [s["step"] for s in data["pipeline"]]
        assert "TRAFFIC_GENERATION" in step_names
        assert "SVM_INFERENCE" in step_names
        assert "RISK_ASSESSMENT" in step_names
        assert "RESPONSE_ENGINE" in step_names

        # For DDoS attack, severity should be HIGH or CRITICAL
        assert data["severity"] in ("HIGH", "CRITICAL"), \
            f"DDoS should produce HIGH/CRITICAL severity, got {data['severity']}"

        # Isolation should be applied for HIGH/CRITICAL
        assert data["isolation_applied"] is True, \
            "DDoS attack should trigger device isolation"
        assert data["is_isolated"] is True

        # Check MongoDB persistence step
        mongo_steps = [s for s in data["pipeline"] if s["step"] == "MONGODB_PERSIST"]
        assert len(mongo_steps) > 0, "MongoDB persistence step missing"
        assert mongo_steps[0]["status"] == "OK", \
            f"MongoDB persist failed: {mongo_steps[0].get('detail')}"

    @pytest.mark.asyncio
    async def test_ganache_transaction(self, client):
        """Verify real Ganache transaction is recorded on-chain and stored in MongoDB."""
        # First, release any prior isolation for ESP32_005
        await client.post("/api/simulate/release", json={"device_id": "ESP32_005"})

        resp = await client.post("/api/simulate/attack", json={
            "device_id": "ESP32_005",
            "attack_type": "DoS",
        })
        assert resp.status_code == 200
        data = resp.json()

        # Check Ganache step
        ganache_steps = [s for s in data["pipeline"] if s["step"] == "GANACHE_BLOCKCHAIN"]
        if ganache_steps:
            step = ganache_steps[0]
            if step["status"] == "OK":
                # Must have real tx hash
                assert data["tx_hash"] is not None
                assert data["tx_hash"].startswith("0x")
                assert len(data["tx_hash"]) == 66  # 0x + 64 hex chars
                assert data["ganache_block"] is not None

                # Verify on-chain via Ganache client
                from blockchain.ganache_client import get_ganache_client
                g = get_ganache_client()
                verification = g.verify_transaction(data["tx_hash"])
                assert verification.get("verified") is True
                assert verification.get("block_number") == data["ganache_block"]

                # Verify transaction and event matching in MongoDB
                import backend.core.mongodb as mongo_mod
                db = mongo_mod.get_mongodb()
                bc_rec = await db.blockchain_records.find_one({"tx_hash": data["tx_hash"]})
                assert bc_rec is not None, "Transaction hash missing from MongoDB blockchain_records"
                assert bc_rec["event_id"] == data["event_id"]

                evt_rec = await db.attack_events.find_one({"event_id": data["event_id"]})
                assert evt_rec is not None, "Event missing from MongoDB attack_events"
                assert evt_rec.get("tx_hash") == data["tx_hash"]
            elif step["status"] == "NOT_CONNECTED":
                pytest.skip("Ganache not running — skipping tx verification")

    @pytest.mark.asyncio
    async def test_telegram_security_alert_delivery(self, client):
        """Verify Telegram security alert is successfully delivered for HIGH/CRITICAL events."""
        from security.telegram_service import is_configured
        if not is_configured():
            pytest.skip("Telegram credentials not configured in environment")

        await client.post("/api/simulate/release", json={"device_id": "ESP32_006"})
        resp = await client.post("/api/simulate/attack", json={
            "device_id": "ESP32_006",
            "attack_type": "DDoS",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "telegram" in data
        assert data["telegram"].get("sent") is True, f"Telegram failed: {data['telegram']}"
        assert "message_id" in data["telegram"]

    @pytest.mark.asyncio
    async def test_device_isolation_and_release(self, client):
        """Test: attack → isolate → blocked → release → normal."""
        device_id = "ESP32_007"

        # Ensure device is not already isolated
        await client.post("/api/simulate/release", json={"device_id": device_id})

        # Attack the device
        resp = await client.post("/api/simulate/attack", json={
            "device_id": device_id,
            "attack_type": "DDoS",
        })
        assert resp.status_code == 200
        attack_data = resp.json()
        assert attack_data["isolation_applied"] is True

        # Verify device is now isolated (second attack should be BLOCKED)
        resp2 = await client.post("/api/simulate/attack", json={
            "device_id": device_id,
            "attack_type": "DDoS",
        })
        assert resp2.status_code == 200
        blocked = resp2.json()
        assert blocked["prediction"] == "ISOLATED_DEVICE"
        assert blocked["is_isolated"] is True

        # Verify device shows isolated in device list
        resp3 = await client.get("/api/simulate/devices")
        devices = resp3.json()
        device = next((d for d in devices if d["device_id"] == device_id), None)
        assert device is not None
        assert device["is_isolated"] is True

        # Release the device
        resp4 = await client.post("/api/simulate/release", json={"device_id": device_id})
        assert resp4.status_code == 200
        release_data = resp4.json()
        assert release_data["status"] == "NORMAL"

        # Verify device is back to normal
        resp5 = await client.get("/api/simulate/devices")
        devices = resp5.json()
        device = next((d for d in devices if d["device_id"] == device_id), None)
        assert device is not None
        assert device["is_isolated"] is False

    @pytest.mark.asyncio
    async def test_isolation_containment_does_not_affect_others(self, client):
        """Isolating one device must NOT isolate other devices."""
        target = "ESP32_008"
        other = "ESP32_009"

        # Ensure both are released
        await client.post("/api/simulate/release", json={"device_id": target})
        await client.post("/api/simulate/release", json={"device_id": other})

        # Attack only the target
        resp = await client.post("/api/simulate/attack", json={
            "device_id": target,
            "attack_type": "DDoS",
        })
        assert resp.status_code == 200

        # Verify other device is NOT isolated
        resp2 = await client.get("/api/simulate/devices")
        devices = resp2.json()
        other_dev = next((d for d in devices if d["device_id"] == other), None)
        assert other_dev is not None
        assert other_dev["is_isolated"] is False, \
            f"Device {other} was incorrectly isolated when {target} was attacked"

        # Clean up
        await client.post("/api/simulate/release", json={"device_id": target})

    @pytest.mark.asyncio
    async def test_all_simulation_devices(self, client):
        """Run attack on ALL simulation devices and verify each works independently."""
        resp = await client.get("/api/simulate/devices")
        assert resp.status_code == 200
        devices = resp.json()
        assert len(devices) >= 1

        # Release all devices first
        for d in devices:
            await client.post("/api/simulate/release", json={"device_id": d["device_id"]})

        # Attack each device
        results = []
        for d in devices:
            resp = await client.post("/api/simulate/attack", json={
                "device_id": d["device_id"],
                "attack_type": "DDoS",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["device_id"] == d["device_id"]
            assert "prediction" in data
            assert "risk_score" in data
            results.append(data)

        # All should have triggered isolation
        isolated_count = sum(1 for r in results if r["isolation_applied"])
        assert isolated_count == len(devices), \
            f"Only {isolated_count}/{len(devices)} devices were isolated"

        # Clean up: release all
        for d in devices:
            await client.post("/api/simulate/release", json={"device_id": d["device_id"]})

    @pytest.mark.asyncio
    async def test_ganache_blockchain_status(self, client):
        """Ganache status endpoint must report real connection state."""
        resp = await client.get("/api/blockchain/ganache/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "connected" in data
        assert "rpc_url" in data

    @pytest.mark.asyncio
    async def test_blockchain_status_uses_ganache(self, client):
        """Primary blockchain status must reference Ganache, not JSON ledger."""
        resp = await client.get("/api/blockchain/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("blockchain") == "ganache", \
            "Primary blockchain status should report 'ganache'"
        assert "ganache" in data
