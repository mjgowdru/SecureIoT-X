"""Unit tests for IoT traffic generator."""
import pytest
from simulation.traffic_generator import TrafficGenerator, DEMO_SCENARIOS


class TestTrafficGenerator:
    """Test IoT traffic simulation data generation."""

    def setup_method(self):
        self.gen = TrafficGenerator(seed=42)

    def test_generate_normal_event(self):
        event = self.gen.generate_normal()
        assert event["device_id"].startswith("ESP32_")
        assert event["label"] == "Normal"
        assert event["packet_count"] > 0
        assert "timestamp" in event

    def test_generate_attack_ddos(self):
        event = self.gen.generate_attack("DDoS")
        assert event["label"] == "DDoS"
        assert event["packet_count"] > 100

    def test_generate_attack_dos(self):
        event = self.gen.generate_attack("DoS")
        assert event["label"] == "DoS"

    def test_generate_suspicious(self):
        event = self.gen.generate_suspicious()
        assert event["packet_count"] > 0

    def test_normal_vs_attack_difference(self):
        normal = self.gen.generate_normal()
        attack = self.gen.generate_attack("DDoS")
        # Attack should have higher packet counts
        assert attack["packet_count"] > normal["packet_count"]

    def test_multiple_events_unique(self):
        events = [self.gen.generate_normal() for _ in range(10)]
        event_ids = [e["event_id"] for e in events]
        assert len(set(event_ids)) == 10

    def test_demo_scenarios_exist(self):
        assert len(DEMO_SCENARIOS) > 0
        for scenario in DEMO_SCENARIOS:
            assert "name" in scenario or isinstance(scenario, str)

    def test_scenario_sequence(self):
        events = self.gen.generate_scenario_sequence("ddos_attack")
        assert len(events) > 0
        assert any(e["label"] == "DDoS" for e in events)

    def test_device_id_override(self):
        event = self.gen.generate_normal("ESP32_007")
        assert event["device_id"] == "ESP32_007"

    def test_event_fields_complete(self):
        event = self.gen.generate_normal()
        required = ["device_id", "event_id", "timestamp", "packet_count", "protocol"]
        for field in required:
            assert field in event, f"Missing field: {field}"
