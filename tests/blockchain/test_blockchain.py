"""Tests for the blockchain — create, verify, tamper, hash integrity."""
import hashlib
import os
import tempfile

import pytest
from blockchain.chain import SecurityBlockchain
from blockchain.block import Block


class TestBlock:
    """Test individual block creation and hashing."""

    def test_block_creation(self):
        block = Block(
            index=1, event_id="EVT_001", device_id="ESP32_001",
            attack_type="DDoS", severity="HIGH", risk_score=85.0,
            action="ISOLATE", previous_hash="0" * 64,
        )
        assert block.index == 1
        assert block.event_id == "EVT_001"
        assert len(block.hash) == 64  # SHA-256

    def test_event_hash_reproducible(self):
        b1 = Block(1, "E1", "D1", "DDoS", "HIGH", 85, "ISOLATE", "0" * 64)
        b2 = Block(1, "E1", "D1", "DDoS", "HIGH", 85, "ISOLATE", "0" * 64)
        assert b1.event_hash == b2.event_hash

    def test_event_hash_changes_on_different_data(self):
        b1 = Block(1, "E1", "D1", "DDoS", "HIGH", 85, "ISOLATE", "0" * 64)
        b2 = Block(1, "E1", "D1", "Normal", "LOW", 10, "ALLOW", "0" * 64)
        assert b1.event_hash != b2.event_hash

    def test_block_to_dict(self):
        block = Block(1, "E1", "D1", "DDoS", "HIGH", 85, "ISOLATE", "0" * 64)
        d = block.to_dict()
        assert d["index"] == 1
        assert "hash" in d
        assert "event_hash" in d
        assert "timestamp" in d


class TestSecurityBlockchain:
    """Test the full blockchain implementation."""

    @pytest.fixture
    def temp_chain(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield SecurityBlockchain(difficulty=1, ledger_path=path)
        os.unlink(path)

    def test_genesis_block(self, temp_chain):
        assert temp_chain.length >= 1
        genesis = temp_chain.chain[0]
        assert genesis.index == 0

    def test_record_event(self, temp_chain):
        block = temp_chain.record_event(
            "EVT_001", "ESP32_001", "DDoS", "CRITICAL", 92.0, "ISOLATE",
        )
        assert block.index == 1
        assert temp_chain.length == 2

    def test_chain_validation(self, temp_chain):
        temp_chain.record_event("E1", "D1", "DDoS", "HIGH", 85, "ISOLATE")
        temp_chain.record_event("E2", "D2", "Normal", "LOW", 10, "ALLOW")
        is_valid, errors = temp_chain.validate_chain()
        assert is_valid is True
        assert len(errors) == 0

    def test_tamper_detection(self, temp_chain):
        temp_chain.record_event("E1", "D1", "DDoS", "HIGH", 85, "ISOLATE")
        temp_chain.record_event("E2", "D2", "Normal", "LOW", 10, "ALLOW")

        # Tamper with block 1
        temp_chain.chain[1].attack_type = "TAMPERED"

        is_valid, errors = temp_chain.validate_chain()
        assert is_valid is False
        assert len(errors) > 0

    def test_hash_chain_linkage(self, temp_chain):
        temp_chain.record_event("E1", "D1", "DDoS", "HIGH", 85, "ISOLATE")
        # Block 1's previous_hash should match block 0's hash
        assert temp_chain.chain[1].previous_hash == temp_chain.chain[0].hash

    def test_persistence(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            # Create and save
            bc1 = SecurityBlockchain(difficulty=1, ledger_path=path)
            bc1.record_event("E1", "D1", "DDoS", "HIGH", 85, "ISOLATE")
            length1 = bc1.length

            # Reload from disk
            bc2 = SecurityBlockchain(difficulty=1, ledger_path=path)
            assert bc2.length == length1
        finally:
            os.unlink(path)

    def test_get_status(self, temp_chain):
        temp_chain.record_event("E1", "D1", "DDoS", "HIGH", 85, "ISOLATE")
        status = temp_chain.get_status()
        assert status["total_blocks"] >= 2
        assert "chain_valid" in status

    def test_proof_of_work(self, temp_chain):
        block = temp_chain.record_event("E1", "D1", "DoS", "HIGH", 70, "ISOLATE")
        # With difficulty 1, hash should start with "0"
        assert block.hash.startswith("0")

    def test_manual_hash_verification(self):
        """Verify SHA-256 hash computation matches manual calculation."""
        event_data = "EVT_001ESP32_003DDoSHIGH85.0ISOLATE"
        expected = hashlib.sha256(event_data.encode()).hexdigest()

        block = Block(1, "EVT_001", "ESP32_003", "DDoS", "HIGH", 85.0, "ISOLATE", "0" * 64)
        assert block.event_hash == expected
