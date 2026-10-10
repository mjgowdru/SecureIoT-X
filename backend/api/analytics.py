"""Analytics API — Dashboard overview stats and AI metrics.

All device/event counts come from MongoDB (the ONLY application database).
"""

import json
from pathlib import Path
from fastapi import APIRouter

from backend.core.config import settings
from backend.core.mongodb import get_mongodb

router = APIRouter()


@router.get("/overview", summary="Dashboard Overview Stats")
async def overview():
    """Get overview statistics for the dashboard from MongoDB."""
    db = get_mongodb()

    total_devices = await db.devices.count_documents({})
    online_devices = await db.devices.count_documents({"is_online": {"$ne": False}})
    isolated_devices = await db.devices.count_documents({"is_isolated": True})
    physical_devices = await db.devices.count_documents({"is_simulation": False})
    simulation_devices = await db.devices.count_documents({"is_simulation": {"$ne": False}})
    total_threats = await db.attack_events.count_documents({})
    critical_threats = await db.attack_events.count_documents({"severity": "CRITICAL"})
    total_predictions = await db.predictions.count_documents({})

    return {
        "total_devices": total_devices,
        "online_devices": online_devices,
        "physical_devices": physical_devices,
        "simulation_devices": simulation_devices,
        "total_threats": total_threats,
        "critical_threats": critical_threats,
        "isolated_devices": isolated_devices,
        "total_predictions": total_predictions,
    }


@router.get("/metrics", summary="AI Model Metrics")
async def ai_metrics():
    """Get latest AI model performance metrics."""
    report_path = Path(settings.project_root) / "reports" / "full_training_report.json"
    if not report_path.exists():
        return {"message": "No training report. Run: python -m ai_engine"}
    with open(report_path) as f:
        return json.load(f)


@router.get("/comparison", summary="Model Comparison")
async def model_comparison():
    """Get model comparison data."""
    report_path = Path(settings.project_root) / "reports" / "model_report.json"
    if not report_path.exists():
        return {"message": "No comparison report. Run training first."}
    with open(report_path) as f:
        return json.load(f)
