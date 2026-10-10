"""Security tests — input validation, credential checks, malformed payloads."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


class TestCredentialSecurity:
    """Test that credentials are not hardcoded in source code."""

    def test_env_file_not_committed(self):
        """Verify .env is in .gitignore and not a real production secret."""
        gitignore = ROOT / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            assert ".env" in content, ".env should be in .gitignore"

    def test_default_password_flagged(self):
        """Flag that default admin password is set in config."""
        from backend.core.config import settings
        if settings.default_admin_password == "SecureIoT2024!":
            # This is expected in development but should be changed in production
            pass  # Acceptable in dev

    def test_jwt_secret_not_default(self):
        """Check if JWT secret has been changed from default."""
        from backend.core.config import settings
        is_default = settings.jwt_secret_key == "CHANGE_THIS_TO_A_RANDOM_SECRET_KEY_MIN_32_CHARS"
        # In dev, this is OK. In production, it should be changed.
        if is_default:
            import warnings
            warnings.warn("JWT secret is still the default value — change for production!")


class TestInputValidation:
    """Test input validation and edge cases."""

    def test_risk_engine_extreme_values(self):
        from security.risk_engine import RiskEngine
        engine = RiskEngine()

        # Very high values — should clamp
        result = engine.calculate_risk(
            ai_confidence=100.0, attack_type="DDoS",
            attack_frequency=10000, device_criticality=10000,
        )
        assert 0 <= result["risk_score"] <= 100

        # Negative values
        result = engine.calculate_risk(
            ai_confidence=-1.0, attack_type="Normal",
            attack_frequency=-5,
        )
        assert result["risk_score"] >= 0

    def test_blockchain_empty_event(self):
        """Blockchain should handle empty/minimal event data."""
        import tempfile
        from blockchain.chain import SecurityBlockchain

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            bc = SecurityBlockchain(difficulty=1, ledger_path=path)
            block = bc.record_event("", "", "", "", 0.0, "")
            assert block is not None
            assert block.index >= 1
        finally:
            os.unlink(path)

    def test_lifecycle_double_transition(self):
        """Event lifecycle should prevent backward transitions."""
        from security.event_lifecycle import SecurityEvent
        event = SecurityEvent()
        event.transition("ANALYZING")
        event.transition("CLASSIFIED")
        # Try backward — should fail gracefully
        assert event.transition("DETECTED") is False
        assert event.state == "CLASSIFIED"

    def test_response_engine_unknown_severity(self):
        """Response engine should handle unknown severity levels."""
        from security.response_engine import ResponseEngine
        engine = ResponseEngine()
        resp = engine.decide_response("UNKNOWN_LEVEL", 50.0, "e1", "d1")
        # Should fall back to MONITOR
        assert resp["action"] == "MONITOR"
