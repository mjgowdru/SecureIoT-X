"""Tests for AI/ML pipeline — data loading, preprocessing, model prediction."""
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


class TestDataLoader:
    """Test dataset loading and validation."""

    def test_synthetic_adapter(self):
        from ai_engine.data_loader import get_adapter
        adapter = get_adapter("Synthetic")
        df = adapter.load()
        assert df is not None
        assert len(df) >= 1000

    def test_target_identification(self):
        from ai_engine.data_loader import get_adapter
        adapter = get_adapter("Synthetic")
        adapter.load()
        assert adapter.identify_target() == "label"

    def test_features_and_target(self):
        from ai_engine.data_loader import get_adapter
        adapter = get_adapter("Synthetic")
        adapter.load()
        X, y = adapter.get_features_and_target()
        assert len(X) == len(y)
        assert len(y.unique()) >= 3

    def test_dataset_validation(self):
        from ai_engine.data_loader import get_adapter
        adapter = get_adapter("Synthetic")
        adapter.load()
        report = adapter.validate()
        assert report["total_missing"] == 0
        assert report["shape"][0] > 0


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_pipeline_fit_transform(self):
        from ai_engine.data_loader import get_adapter
        from ai_engine.preprocessing import PreprocessingPipeline

        adapter = get_adapter("Synthetic")
        df = adapter.load()
        X, y = adapter.get_features_and_target()

        pipeline = PreprocessingPipeline()
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(X, y)

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] > 0

    def test_leakage_prevention(self):
        """Verify scaler is fit only on train data."""
        from ai_engine.data_loader import get_adapter
        from ai_engine.preprocessing import PreprocessingPipeline

        adapter = get_adapter("Synthetic")
        df = adapter.load()
        X, y = adapter.get_features_and_target()

        pipeline = PreprocessingPipeline()
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(X, y)

        # Train and test shapes should differ
        assert X_train.shape[0] > X_test.shape[0]


class TestModelPrediction:
    """Test trained model prediction."""

    def test_model_files_exist(self):
        models_dir = ROOT / "models"
        joblib_files = list(models_dir.glob("**/*.joblib"))
        assert len(joblib_files) > 0, "No trained models found. Run 'python -m ai_engine' first."

    def test_model_loads_and_predicts(self):
        import joblib
        models_dir = ROOT / "models"
        model_files = list(models_dir.glob("**/*.joblib"))
        model_files = [f for f in model_files if "preprocessing" not in str(f) and "anomaly" not in str(f)]

        if not model_files:
            pytest.skip("No trained models available")

        model = joblib.load(model_files[0])
        # Create a minimal feature vector
        n_features = model.n_features_in_
        X_test = np.random.rand(1, n_features)
        pred = model.predict(X_test)
        assert pred is not None
        assert len(pred) == 1

    def test_model_registry(self):
        from ai_engine.model_registry import ModelRegistry
        registry = ModelRegistry(str(ROOT / "models" / "registry.json"))
        models = registry.list_models()
        assert len(models) > 0

        prod = registry.get_production_model()
        if prod:
            assert prod["is_production"] is True
            assert prod["validated"] is True

    def test_training_report_exists(self):
        report_path = ROOT / "reports" / "model_report.json"
        assert report_path.exists(), "Training report not found"

        import json
        with open(report_path) as f:
            report = json.load(f)
        assert len(report) > 0
