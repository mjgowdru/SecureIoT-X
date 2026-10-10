"""Unit tests for the Response Engine."""
import pytest
from security.response_engine import ResponseEngine


class TestResponseEngine:
    """Test automated response decision logic."""

    def setup_method(self):
        self.engine = ResponseEngine()

    def test_low_risk_allows(self):
        resp = self.engine.decide_response("LOW", 15.0, "evt-001", "ESP32_001")
        assert resp["action"] == "ALLOW"
        assert resp["should_alert"] is False

    def test_medium_risk_monitors(self):
        resp = self.engine.decide_response("MEDIUM", 45.0, "evt-002", "ESP32_002")
        assert resp["action"] == "MONITOR"

    def test_high_risk_isolates(self):
        resp = self.engine.decide_response("HIGH", 75.0, "evt-003", "ESP32_003")
        assert resp["action"] == "ISOLATE"

    def test_critical_risk_full_response(self):
        resp = self.engine.decide_response("CRITICAL", 95.0, "evt-004", "ESP32_004")
        assert resp["action"] == "ISOLATE"
        assert resp["should_alert"] is True
        assert resp["should_block"] is True

    def test_response_has_required_fields(self):
        resp = self.engine.decide_response("HIGH", 75.0, "evt-005", "ESP32_005", "DDoS")
        required = ["response_id", "event_id", "device_id", "action", "severity",
                     "risk_score", "timestamp", "status"]
        for field in required:
            assert field in resp, f"Missing: {field}"

    def test_execute_response_simulated(self):
        resp = self.engine.decide_response("HIGH", 80.0, "evt-006", "ESP32_006")
        result = self.engine.execute_response(resp)
        assert result["status"] == "SUCCESS"
        assert result["sdn_mode"] == "mock"

    def test_response_id_unique(self):
        r1 = self.engine.decide_response("LOW", 10, "evt-a", "d-1")
        r2 = self.engine.decide_response("LOW", 10, "evt-b", "d-1")
        assert r1["response_id"] != r2["response_id"]
