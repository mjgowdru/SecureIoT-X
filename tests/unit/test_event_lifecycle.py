"""Unit tests for the Event Lifecycle State Machine."""
import pytest
from security.event_lifecycle import SecurityEvent, EventLifecycleManager, LIFECYCLE_STATES


class TestSecurityEvent:
    """Test the event lifecycle state machine."""

    def test_initial_state(self):
        event = SecurityEvent(device_id="ESP32_001")
        assert event.state == "DETECTED"
        assert "DETECTED" in event.timestamps

    def test_valid_forward_transition(self):
        event = SecurityEvent()
        assert event.transition("ANALYZING") is True
        assert event.state == "ANALYZING"
        assert event.transition("CLASSIFIED") is True
        assert event.state == "CLASSIFIED"

    def test_backward_transition_fails(self):
        event = SecurityEvent()
        event.transition("ANALYZING")
        event.transition("CLASSIFIED")
        assert event.transition("DETECTED") is False
        assert event.state == "CLASSIFIED"

    def test_invalid_state_fails(self):
        event = SecurityEvent()
        assert event.transition("INVALID_STATE") is False

    def test_full_lifecycle(self):
        event = SecurityEvent()
        for state in LIFECYCLE_STATES[1:]:  # Skip DETECTED (initial)
            assert event.transition(state) is True
        assert event.state == "RESOLVED"
        assert event.is_resolved

    def test_set_data(self):
        event = SecurityEvent()
        event.set_data("prediction", "DDoS")
        assert event.data["prediction"] == "DDoS"

    def test_to_dict(self):
        event = SecurityEvent(device_id="ESP32_003")
        d = event.to_dict()
        assert d["device_id"] == "ESP32_003"
        assert "event_id" in d
        assert "state" in d


class TestEventLifecycleManager:
    """Test lifecycle manager for multiple events."""

    def setup_method(self):
        self.manager = EventLifecycleManager()

    def test_create_event(self):
        event = self.manager.create_event("ESP32_001")
        assert event.state == "DETECTED"
        assert self.manager.get_event(event.event_id) is not None

    def test_transition_event(self):
        event = self.manager.create_event("ESP32_002")
        self.manager.transition_event(event.event_id, "ANALYZING")
        assert event.state == "ANALYZING"

    def test_process_full_lifecycle(self):
        event = self.manager.process_full_lifecycle(
            device_id="ESP32_003",
            prediction={"prediction": "DDoS", "confidence": 0.95, "model_name": "RF"},
            risk={"risk_score": 88.5, "severity": "CRITICAL"},
            response={"action": "ISOLATE", "status": "SUCCESS"},
            blockchain_block=5,
        )
        assert event.state == "VERIFIED"
        assert event.data["prediction"] == "DDoS"
        assert event.data["blockchain_block"] == 5
        assert len(event.timestamps) >= 7

    def test_active_events(self):
        e1 = self.manager.create_event("ESP32_001")
        e2 = self.manager.create_event("ESP32_002")
        assert len(self.manager.get_active_events()) == 2

    def test_nonexistent_event_transition(self):
        assert self.manager.transition_event("nonexistent", "ANALYZING") is False
