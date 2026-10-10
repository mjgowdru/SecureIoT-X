"""Shared test fixtures and path setup for SecureIoT-X tests."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for all tests
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
