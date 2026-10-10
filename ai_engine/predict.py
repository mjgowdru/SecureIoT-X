"""
SecureIoT-X AI Engine — Prediction / Inference Engine.

Loads a trained model + preprocessing pipeline and produces predictions
with confidence scores, risk assessment, and recommended actions.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("secureiot.ai.predict")


class PredictionEngine:
    """
    Production inference engine.

    Loads a saved model + preprocessing pipeline and produces structured
    predictions with confidence, severity, and explainability.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessing_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.model = None
        self.preprocessing = None
        self.model_name: str = model_name or "unknown"
        self.model_version: str = "unknown"
        self.config = config or {}
        self._min_confidence = self.config.get("min_confident", 0.60)

        # Auto-discover model_path / preprocessing_path if not provided
        if not model_path:
            reg_path = Path("models/registry.json")
            if reg_path.exists():
                try:
                    from ai_engine.model_registry import ModelRegistry
                    registry = ModelRegistry(str(reg_path))
                    if model_name:
                        matching = [m for m in registry.list_models() if m.get("model_name", "").lower() == model_name.lower()]
                        if matching:
                            target = matching[-1]
                            model_path = target.get("model_path")
                            if not preprocessing_path:
                                preprocessing_path = target.get("preprocessing_path")
                    else:
                        prod = registry.get_production_model()
                        if prod:
                            model_path = prod.get("model_path")
                            if not preprocessing_path:
                                preprocessing_path = prod.get("preprocessing_path")
                except Exception as e:
                    logger.debug("Registry lookup skipped: %s", e)

            # Search by model_name in models dir
            if not model_path and model_name:
                candidates = list(Path("models").glob(f"**/{model_name}*.joblib"))
                if candidates:
                    model_path = str(candidates[0])

            # If still None, look for any model in models/
            if not model_path:
                candidates = [
                    f for f in Path("models").glob("**/*.joblib")
                    if "preprocessing" not in str(f) and "anomaly" not in str(f)
                ]
                if candidates:
                    model_path = str(candidates[0])

        if not preprocessing_path:
            default_pp = Path("models/preprocessing")
            if (default_pp / "preprocessing_pipeline.joblib").exists():
                preprocessing_path = str(default_pp)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        if preprocessing_path and (Path(preprocessing_path) / "preprocessing_pipeline.joblib").exists():
            self.load_preprocessing(preprocessing_path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        p = Path(path)
        parts = p.stem.split("_", 1)
        self.model_name = parts[0]
        self.model_version = parts[1] if len(parts) > 1 else "v1"
        logger.info("Loaded model: %s (version: %s) from %s", self.model_name, self.model_version, path)

    def load_preprocessing(self, path: str) -> None:
        """Load preprocessing pipeline."""
        from ai_engine.preprocessing import PreprocessingPipeline
        self.preprocessing = PreprocessingPipeline.load(path)
        logger.info("Loaded preprocessing pipeline from %s", path)

    def predict(
        self,
        features: Dict[str, Any],
        device_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Make a single prediction from raw feature dict.

        Returns structured prediction with confidence, risk, severity.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Preprocess if pipeline available
        if self.preprocessing is not None:
            try:
                X = self.preprocessing.transform(df)
            except Exception as e:
                logger.warning("Preprocessing failed, using raw features: %s", e)
                X = df.select_dtypes(include=[np.number]).values
        else:
            X = df.select_dtypes(include=[np.number]).values

        # Predict with timing
        start = time.perf_counter()
        y_pred = self.model.predict(X)
        inference_ms = (time.perf_counter() - start) * 1000

        # Get confidence
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            confidence = float(np.max(proba))
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            # Normalize to 0-1 range
            confidence = float(1.0 / (1.0 + np.exp(-np.abs(decision).max())))

        # Decode label
        pred_label = y_pred[0]
        if self.preprocessing and self.preprocessing._target_encoder:
            try:
                pred_label = self.preprocessing.inverse_transform_target(y_pred)[0]
            except Exception:
                pred_label = str(y_pred[0])
        else:
            pred_label = str(pred_label)

        # Check confidence threshold
        if confidence < self._min_confidence:
            pred_label = "SUSPICIOUS_UNKNOWN"

        # Build result
        result = {
            "prediction_id": uuid.uuid4().hex[:16],
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "model": self.model_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device_id": device_id,
            "inference_time_ms": round(inference_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return result

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        device_id: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """Predict on multiple feature dicts."""
        return [self.predict(f, device_id) for f in features_list]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
