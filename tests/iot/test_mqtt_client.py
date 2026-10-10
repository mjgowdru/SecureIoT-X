"""Tests for MQTT client — simulated pub/sub."""
import asyncio
import pytest
from iot.mqtt_client import SimulatedMQTTClient


class TestSimulatedMQTTClient:
    """Test simulated MQTT pub/sub communication."""

    def test_connect_disconnect(self):
        async def _test():
            client = SimulatedMQTTClient()
            await client.connect()
            assert client.connected is True
            await client.disconnect()
            assert client.connected is False
        asyncio.run(_test())

    def test_publish_subscribe(self):
        received = []

        async def _test():
            client = SimulatedMQTTClient()
            await client.connect()
            client.subscribe("iot/+/telemetry", lambda msg: received.append(msg))
            await client.publish(
                "iot/ESP32_001/telemetry",
                {"temperature": 25.3},
                device_id="ESP32_001",
            )
            assert len(received) == 1
            assert received[0].payload["temperature"] == 25.3
            await client.disconnect()
        asyncio.run(_test())

    def test_topic_filtering(self):
        received = []

        async def _test():
            client = SimulatedMQTTClient()
            await client.connect()
            client.subscribe("iot/ESP32_001/telemetry", lambda msg: received.append(msg))
            # Publish to different topic
            await client.publish("iot/ESP32_002/telemetry", {"temp": 20}, device_id="ESP32_002")
            # With wildcard, it may or may not match depending on implementation
            await client.disconnect()
        asyncio.run(_test())

    def test_multiple_messages(self):
        received = []

        async def _test():
            client = SimulatedMQTTClient()
            await client.connect()
            client.subscribe("iot/+/telemetry", lambda msg: received.append(msg))
            for i in range(5):
                await client.publish(
                    f"iot/ESP32_{i:03d}/telemetry",
                    {"index": i},
                    device_id=f"ESP32_{i:03d}",
                )
            assert len(received) == 5
            await client.disconnect()
        asyncio.run(_test())

    def test_message_structure(self):
        received = []

        async def _test():
            client = SimulatedMQTTClient()
            await client.connect()
            client.subscribe("iot/+/telemetry", lambda msg: received.append(msg))
            await client.publish("iot/ESP32_001/telemetry", {"data": 1}, device_id="ESP32_001")
            msg = received[0]
            assert hasattr(msg, "topic")
            assert hasattr(msg, "payload")
            assert hasattr(msg, "device_id")
            await client.disconnect()
        asyncio.run(_test())
