"""Unit tests for the Risk Engine."""
import pytest
from security.risk_engine import RiskEngine


class TestRiskEngine:
    """Test configurable risk scoring."""

    def setup_method(self):
        self.engine = RiskEngine()

    def test_normal_traffic_low_risk(self):
        result = self.engine.calculate_risk(ai_confidence=0.1, attack_type="Normal")
        assert result["severity"] == "LOW"
        assert result["risk_score"] < 30

    def test_ddos_high_risk(self):
        result = self.engine.calculate_risk(
            ai_confidence=0.95, attack_type="DDoS",
            attack_frequency=5, device_criticality=90,
        )
        assert result["severity"] in ("HIGH", "CRITICAL")
        assert result["risk_score"] > 60

    def test_critical_threshold(self):
        result = self.engine.calculate_risk(
            ai_confidence=0.99, attack_type="DDoS",
            attack_frequency=10, device_criticality=100,
            anomaly_score=-3.0, threat_history_count=7,
        )
        assert result["severity"] == "CRITICAL"
        assert result["risk_score"] > 80

    def test_risk_score_clamped(self):
        result = self.engine.calculate_risk(
            ai_confidence=5.0,  # Intentionally high
            attack_type="DDoS",
            attack_frequency=100,
        )
        assert 0 <= result["risk_score"] <= 100

    def test_risk_factors_returned(self):
        result = self.engine.calculate_risk(ai_confidence=0.5, attack_type="DoS")
        assert "factors" in result
        factors = result["factors"]
        assert "ai_confidence" in factors
        assert "attack_severity" in factors

    def test_unknown_attack_type(self):
        result = self.engine.calculate_risk(ai_confidence=0.8, attack_type="UnknownType")
        # Should not crash, should use default score
        assert result["risk_score"] > 0

    def test_classify_severity(self):
        assert self.engine.classify_severity(10) == "LOW"
        assert self.engine.classify_severity(45) == "MEDIUM"
        assert self.engine.classify_severity(70) == "HIGH"
        assert self.engine.classify_severity(95) == "CRITICAL"
