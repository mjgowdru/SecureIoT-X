"""End-to-end tests — full flow from IoT simulation to blockchain verification."""
import os
import tempfile
import time

import pytest
from simulation.traffic_generator import TrafficGenerator
from iot.mqtt_client import SimulatedMQTTClient, MQTTMessage
from iot.telemetry_handler import TelemetryHandler
from blockchain.chain import SecurityBlockchain
from security.event_lifecycle import EventLifecycleManager


class TestEndToEnd:
    """Full end-to-end pipeline verification."""

    @pytest.fixture
    def system(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        bc = SecurityBlockchain(difficulty=1, ledger_path=path)
        handler = TelemetryHandler(blockchain=bc)
        gen = TrafficGenerator(seed=int(time.time()))
        lifecycle = EventLifecycleManager()
        yield {
            "gen": gen, "handler": handler,
            "blockchain": bc, "lifecycle": lifecycle, "path": path,
        }
        os.unlink(path)

    def test_e2e_normal_flow(self, system):
        """Normal IoT traffic should flow through without blockchain entry."""
        event = system["gen"].generate_normal()
        msg = MQTTMessage(
            topic=f"iot/{event['device_id']}/telemetry",
            payload=event, device_id=event["device_id"],
        )
        result = system["handler"].process_message(msg)
        assert result["risk"]["severity"] == "LOW"
        assert result["response"]["action"] == "ALLOW"

    def test_e2e_attack_flow(self, system):
        """Attack traffic should trigger full pipeline including blockchain."""
        event = system["gen"].generate_attack("DDoS", "ESP32_003")
        msg = MQTTMessage(
            topic=f"iot/{event['device_id']}/telemetry",
            payload=event, device_id=event["device_id"],
        )
        result = system["handler"].process_message(msg)

        assert result["risk"]["severity"] in ("HIGH", "CRITICAL")
        assert result["response"]["action"] in ("ISOLATE", "BLOCK")
        assert "blockchain_block" in result

        # Verify blockchain integrity after recording
        is_valid, errors = system["blockchain"].validate_chain()
        assert is_valid, f"Chain invalid after E2E: {errors}"

    def test_e2e_mixed_traffic(self, system):
        """Process mixed normal and attack traffic."""
        events_processed = 0
        attacks_detected = 0

        for _ in range(5):
            event = system["gen"].generate_normal()
            msg = MQTTMessage(
                topic=f"iot/{event['device_id']}/telemetry",
                payload=event, device_id=event["device_id"],
            )
            system["handler"].process_message(msg)
            events_processed += 1

        for attack_type in ["DDoS", "DoS", "Reconnaissance"]:
            event = system["gen"].generate_attack(attack_type)
            msg = MQTTMessage(
                topic=f"iot/{event['device_id']}/telemetry",
                payload=event, device_id=event["device_id"],
            )
            result = system["handler"].process_message(msg)
            events_processed += 1
            if result["risk"]["severity"] in ("HIGH", "CRITICAL"):
                attacks_detected += 1

        stats = system["handler"].get_stats()
        assert stats["total_events"] == events_processed
        assert attacks_detected >= 2

    def test_e2e_blockchain_tamper_after_recording(self, system):
        """Record events then tamper — verify detection."""
        for i in range(3):
            event = system["gen"].generate_attack("DDoS", f"ESP32_{i+1:03d}")
            msg = MQTTMessage(
                topic=f"iot/{event['device_id']}/telemetry",
                payload=event, device_id=event["device_id"],
            )
            system["handler"].process_message(msg)

        # Valid chain
        is_valid, _ = system["blockchain"].validate_chain()
        assert is_valid

        # Tamper with block 1
        if system["blockchain"].length > 1:
            system["blockchain"].chain[1].attack_type = "TAMPERED"
            is_valid, errors = system["blockchain"].validate_chain()
            assert not is_valid

    def test_e2e_lifecycle_tracking(self, system):
        """Verify event lifecycle state transitions."""
        event = system["gen"].generate_attack("DDoS")
        msg = MQTTMessage(
            topic=f"iot/{event['device_id']}/telemetry",
            payload=event, device_id=event["device_id"],
        )
        result = system["handler"].process_message(msg)

        # Create lifecycle event
        lc = system["lifecycle"].process_full_lifecycle(
            device_id=event["device_id"],
            prediction=result["prediction"],
            risk=result["risk"],
            response=result["response"],
            blockchain_block=result.get("blockchain_block"),
        )

        assert lc.state == "VERIFIED"
        assert len(lc.timestamps) >= 7
        assert lc.data.get("severity") in ("HIGH", "CRITICAL")
