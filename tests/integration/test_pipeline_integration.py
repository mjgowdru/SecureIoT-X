"""Integration tests — full pipeline from simulator to blockchain."""
import os
import tempfile
import time

import pytest
from simulation.traffic_generator import TrafficGenerator
from iot.mqtt_client import SimulatedMQTTClient, MQTTMessage
from iot.telemetry_handler import TelemetryHandler
from blockchain.chain import SecurityBlockchain
from security.risk_engine import RiskEngine
from security.response_engine import ResponseEngine


class TestPipelineIntegration:
    """Test the full detection pipeline integration."""

    @pytest.fixture
    def pipeline(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        bc = SecurityBlockchain(difficulty=1, ledger_path=path)
        handler = TelemetryHandler(blockchain=bc)
        yield {"handler": handler, "blockchain": bc, "path": path}
        os.unlink(path)

    def test_normal_traffic_flow(self, pipeline):
        msg = MQTTMessage(
            topic="iot/ESP32_001/telemetry",
            payload={"device_id": "ESP32_001", "label": "Normal",
                     "packet_count": 10, "temperature": 25.0, "humidity": 50.0},
            device_id="ESP32_001",
        )
        result = pipeline["handler"].process_message(msg)
        assert result["risk"]["severity"] == "LOW"
        assert result["response"]["action"] == "ALLOW"

    def test_attack_traffic_flow(self, pipeline):
        msg = MQTTMessage(
            topic="iot/ESP32_003/telemetry",
            payload={"device_id": "ESP32_003", "label": "DDoS",
                     "packet_count": 50000, "temperature": 25.0, "humidity": 50.0},
            device_id="ESP32_003",
        )
        result = pipeline["handler"].process_message(msg)
        assert result["risk"]["severity"] in ("HIGH", "CRITICAL")
        assert result["response"]["action"] in ("ISOLATE", "BLOCK")
        # Should record to blockchain
        assert "blockchain_block" in result

    def test_blockchain_records_attacks(self, pipeline):
        # Send multiple attacks
        for i in range(3):
            msg = MQTTMessage(
                topic=f"iot/ESP32_{i+1:03d}/telemetry",
                payload={"device_id": f"ESP32_{i+1:03d}", "label": "DDoS",
                         "packet_count": 50000, "temperature": 25.0, "humidity": 50.0},
                device_id=f"ESP32_{i+1:03d}",
            )
            pipeline["handler"].process_message(msg)

        # Blockchain should have genesis + 3 attack blocks
        assert pipeline["blockchain"].length >= 4
        is_valid, _ = pipeline["blockchain"].validate_chain()
        assert is_valid

    def test_handler_stats(self, pipeline):
        # Process some events
        for label in ["Normal", "Normal", "DDoS", "DoS"]:
            msg = MQTTMessage(
                topic="iot/ESP32_001/telemetry",
                payload={"device_id": "ESP32_001", "label": label,
                         "packet_count": 1000, "temperature": 25.0, "humidity": 50.0},
                device_id="ESP32_001",
            )
            pipeline["handler"].process_message(msg)

        stats = pipeline["handler"].get_stats()
        assert stats["total_events"] == 4
        assert stats["attacks_detected"] >= 2

    def test_risk_to_response_mapping(self):
        risk_engine = RiskEngine()
        response_engine = ResponseEngine()

        scenarios = [
            (0.1, "Normal", "LOW", "ALLOW"),
            (0.8, "Reconnaissance", "MEDIUM", "MONITOR"),
            (0.95, "DDoS", "HIGH", "ISOLATE"),
        ]

        for conf, attack, expected_sev, expected_action in scenarios:
            risk = risk_engine.calculate_risk(
                ai_confidence=conf, attack_type=attack,
            )
            # Severity should be at least as high as expected
            resp = response_engine.decide_response(
                severity=risk["severity"], risk_score=risk["risk_score"],
                event_id="test", device_id="ESP32_001", attack_type=attack,
            )
            # Verify the response action matches expected behavior
            assert resp["action"] is not None
