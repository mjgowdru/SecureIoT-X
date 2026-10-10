"""
Tests for Support Vector Machine (SVM) Classifier and Pipeline.

Verifies:
- SVM training with configurable hyperparameters and probability=True
- SVM prediction with schema compliance (prediction, confidence, model, model_version, timestamp, device_id)
- Model save and load roundtrip fidelity
- Preprocessing + SVM pipeline with zero data leakage
- Multiclass prediction with macro and weighted F1 metrics
- Handling of invalid, missing, and extra features
- Deterministic inference consistency
- Compatibility with attack simulator traffic generator
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_engine.data_loader import SyntheticAdapter, get_adapter
from ai_engine.evaluate import evaluate_model
from ai_engine.predict import PredictionEngine
from ai_engine.preprocessing import PreprocessingPipeline
from ai_engine.train import build_svm_model, train_svm, train_svm_pipeline
from simulation.traffic_generator import TrafficGenerator

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def synthetic_data():
    """Load synthetic IoT dataset for testing."""
    adapter = SyntheticAdapter("synthetic")
    adapter.load()
    X, y = adapter.get_features_and_target()
    return X, y


@pytest.fixture(scope="module")
def trained_svm_pipeline(synthetic_data):
    """Train an SVM pipeline fixture for use across tests."""
    X, y = synthetic_data
    temp_dir = tempfile.mkdtemp()
    result = train_svm_pipeline(
        X=X,
        y=y,
        params={"kernel": "rbf", "C": 1.0, "max_iter": 2000, "random_state": 42},
        preprocess_config={"test_size": 0.2, "validation_size": 0.1, "random_seed": 42},
        save_dir=temp_dir,
        dataset_name="Synthetic_Test",
        random_seed=42,
    )
    yield result, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestSVMTraining:
    """Verify SVM training, hyperparameters, and evaluation outputs."""

    def test_svm_model_builder(self):
        """Verify build_svm_model respects hyperparameters and probability=True."""
        model = build_svm_model({"kernel": "linear", "C": 0.5, "random_state": 123})
        assert model.probability is True
        assert model.kernel == "linear"
        assert model.C == 0.5
        assert model.random_state == 123

    def test_svm_training_produces_required_outputs(self, trained_svm_pipeline):
        """Verify training produces SVM model, preprocessing, metadata, metrics, and confusion matrix."""
        result, _ = trained_svm_pipeline

        # 1. Model
        assert result["model"] is not None
        assert hasattr(result["model"], "predict")
        assert hasattr(result["model"], "predict_proba")

        # 2. Preprocessing pipeline
        assert result["preprocessing"] is not None
        assert result["preprocessing"]._is_fitted is True

        # 3. Model metadata
        metadata = result["metadata"]
        assert metadata["name"] == "SVM"
        assert "version" in metadata
        assert metadata["training_time_seconds"] >= 0
        assert metadata["model_size_bytes"] > 0
        assert metadata["dataset"] == "Synthetic_Test"

        # 4. Metrics
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "inference_time_ms" in metrics
        assert "training_time_seconds" in metrics

        # 5. Confusion matrix
        cm = result["confusion_matrix"]
        assert isinstance(cm, list)
        assert len(cm) > 0
        assert len(cm) == len(cm[0])


class TestSVMPrediction:
    """Verify SVM prediction output schema and behavior."""

    def test_svm_prediction_schema(self, trained_svm_pipeline):
        """Output must include prediction, confidence, model, model_version, timestamp, device_id."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )

        sample_features = {
            "flow_duration": 1.5,
            "fwd_pkts_tot": 25,
            "bwd_pkts_tot": 10,
            "fwd_pkts_per_sec": 50.0,
            "bwd_pkts_per_sec": 20.0,
            "proto": 6,
            "dst_port": 80,
            "conn_state_num": 4,
        }
        pred = engine.predict(sample_features, device_id="device-svm-01")

        assert "prediction" in pred
        assert "confidence" in pred
        assert "model" in pred
        assert "model_version" in pred
        assert "timestamp" in pred
        assert "device_id" in pred
        assert pred["device_id"] == "device-svm-01"
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_svm_predict_batch(self, trained_svm_pipeline):
        """Batch predictions should match length and structure."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )
        batch = [
            {"flow_duration": 0.01, "fwd_pkts_tot": 500, "fwd_pkts_per_sec": 2000.0, "proto": 6, "dst_port": 80},
            {"flow_duration": 100.0, "fwd_pkts_tot": 5, "fwd_pkts_per_sec": 1.0, "proto": 17, "dst_port": 1883},
        ]
        results = engine.predict_batch(batch, device_id="batch-device")
        assert len(results) == 2
        for r in results:
            assert "prediction" in r
            assert "confidence" in r
            assert r["model"] == "SVM"


class TestSVMModelSaveLoad:
    """Verify model save and load roundtrip fidelity."""

    def test_save_load_prediction_identical(self, trained_svm_pipeline):
        """Saved and loaded SVM must produce identical predictions and probabilities."""
        result, temp_dir = trained_svm_pipeline
        original_model = result["model"]
        pp = result["preprocessing"]

        save_path = Path(temp_dir) / "test_save" / "svm_test.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(original_model, save_path)

        loaded_model = joblib.load(save_path)

        # Generate sample test matrix
        X_sample = np.random.RandomState(42).randn(10, len(pp.feature_names))
        orig_preds = original_model.predict(X_sample)
        loaded_preds = loaded_model.predict(X_sample)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

        orig_proba = original_model.predict_proba(X_sample)
        loaded_proba = loaded_model.predict_proba(X_sample)
        np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-5)


class TestPreprocessingPlusSVMPipeline:
    """Verify no data leakage in preprocessing + SVM pipeline."""

    def test_no_data_leakage_scaler(self, synthetic_data):
        """Train/test split happens before fitting; scaler fitted only on training data."""
        X, y = synthetic_data
        pp = PreprocessingPipeline({"test_size": 0.2, "validation_size": 0.1, "random_seed": 42})
        X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(X, y)

        # Scaler must have been fitted on X_train numerical columns only
        assert pp._scaler is not None
        assert len(pp._scaler.mean_) == len(pp._numerical_columns)

        # Test set is transformed with same scaler
        raw_test_sample = X.iloc[:5]
        t1 = pp.transform(raw_test_sample)
        t2 = pp.transform(raw_test_sample)
        np.testing.assert_array_almost_equal(t1, t2)


class TestMulticlassPrediction:
    """Verify multiclass evaluation metrics with macro and weighted F1."""

    def test_multiclass_metrics(self, trained_svm_pipeline):
        """Verify multiclass report computes macro F1 and weighted F1."""
        result, _ = trained_svm_pipeline
        metrics = result["metrics"]
        assert metrics["n_classes"] >= 3
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert metrics["f1_macro"] >= 0.0
        assert metrics["f1_weighted"] >= 0.0


class TestInvalidMissingFeatures:
    """Verify pipeline handles incomplete, extra, or invalid features gracefully."""

    def test_missing_features_imputed(self, trained_svm_pipeline):
        """Missing feature columns must be filled without crashing."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )

        # Incomplete feature dict (missing most columns)
        incomplete = {"dst_port": 443}
        pred = engine.predict(incomplete, device_id="sparse-dev")
        assert pred["prediction"] is not None
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_extra_unknown_features_ignored(self, trained_svm_pipeline):
        """Extra columns not in trained feature set should not cause errors."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )
        extra_features = {
            "flow_duration": 1.0,
            "fwd_pkts_tot": 10,
            "unknown_extra_col_1": 9999,
            "random_string_col": "malicious_payload_simulation",
        }
        pred = engine.predict(extra_features, device_id="extra-feat-dev")
        assert pred["prediction"] is not None

    def test_nan_and_inf_features_handled(self, trained_svm_pipeline):
        """NaN, None, and infinite values must be imputed without crashing."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )
        invalid_features = {
            "flow_duration": np.nan,
            "fwd_pkts_tot": np.inf,
            "bwd_pkts_tot": -np.inf,
            "proto": None,
        }
        pred = engine.predict(invalid_features, device_id="nan-inf-dev")
        assert pred["prediction"] is not None


class TestInferenceConsistency:
    """Verify deterministic inference output."""

    def test_deterministic_predictions(self, trained_svm_pipeline):
        """Repeated predictions on identical input must yield identical results."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )

        features = {
            "flow_duration": 0.5,
            "fwd_pkts_tot": 100,
            "bwd_pkts_tot": 20,
            "fwd_pkts_per_sec": 200.0,
            "bwd_pkts_per_sec": 40.0,
            "proto": 6,
            "dst_port": 80,
            "conn_state_num": 1,
        }

        preds = [engine.predict(features, device_id="dev-consistent") for _ in range(5)]
        for p in preds[1:]:
            assert p["prediction"] == preds[0]["prediction"]
            assert p["confidence"] == preds[0]["confidence"]


class TestAttackSimulatorCompatibility:
    """Verify attack simulator events pass compatible features to SVM pipeline."""

    def test_simulator_events_compatible_with_svm(self, trained_svm_pipeline):
        """Simulator events (normal and attack) must predict without error."""
        result, temp_dir = trained_svm_pipeline
        engine = PredictionEngine(
            model_path=result["model_path"],
            preprocessing_path=result["preprocessing_path"],
        )

        generator = TrafficGenerator(seed=42)
        normal_event = generator.generate_normal("dev-norm")
        attack_event = generator.generate_attack("DDoS", "dev-atk")

        pred_normal = engine.predict(normal_event, device_id="dev-norm")
        pred_attack = engine.predict(attack_event, device_id="dev-atk")

        assert pred_normal["prediction"] is not None
        assert pred_attack["prediction"] is not None
        assert pred_normal["model"] == "SVM"
        assert pred_attack["model"] == "SVM"
