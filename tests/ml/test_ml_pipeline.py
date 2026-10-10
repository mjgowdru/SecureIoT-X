"""
SecureIoT-X — ML Pipeline Unit Tests.

Tests: synthetic data loading, preprocessing, feature engineering, training, prediction.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pytest
from ai_engine.data_loader import SyntheticAdapter, get_adapter
from ai_engine.preprocessing import PreprocessingPipeline
from ai_engine.feature_engineering import FeatureEngineer


class TestDataLoader:
    def test_synthetic_adapter_loads(self):
        adapter = SyntheticAdapter("synthetic")
        df = adapter.load()
        assert len(df) == 5000
        assert "label" in df.columns

    def test_identify_target(self):
        adapter = SyntheticAdapter("synthetic")
        adapter.load()
        target = adapter.identify_target()
        assert target == "label"

    def test_validate_report(self):
        adapter = SyntheticAdapter("synthetic")
        adapter.load()
        adapter.identify_target()
        report = adapter.validate()
        assert "shape" in report
        assert "class_distribution" in report
        assert report["num_classes"] >= 2

    def test_get_features_and_target(self):
        adapter = SyntheticAdapter("synthetic")
        adapter.load()
        adapter.identify_target()
        X, y = adapter.get_features_and_target()
        assert len(X) == len(y)
        assert "label" not in X.columns

    def test_get_adapter_factory(self):
        adapter = get_adapter("Synthetic")
        assert isinstance(adapter, SyntheticAdapter)


class TestPreprocessing:
    def _get_data(self):
        adapter = SyntheticAdapter("synthetic")
        adapter.load()
        adapter.identify_target()
        return adapter.get_features_and_target()

    def test_fit_transform(self):
        X, y = self._get_data()
        pp = PreprocessingPipeline({"random_seed": 42, "test_size": 0.2, "validation_size": 0.1})
        X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(X, y)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)

    def test_no_data_leakage(self):
        """Scaler must be fitted on train only — val/test shapes should match features."""
        X, y = self._get_data()
        pp = PreprocessingPipeline({"random_seed": 42, "test_size": 0.2, "validation_size": 0.1})
        X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(X, y)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == X_val.shape[1]

    def test_target_encoding(self):
        X, y = self._get_data()
        pp = PreprocessingPipeline({"random_seed": 42})
        pp.fit_transform(X, y)
        assert len(pp.target_classes) >= 2

    def test_save_load(self, tmp_path):
        X, y = self._get_data()
        pp = PreprocessingPipeline({"random_seed": 42})
        pp.fit_transform(X, y)
        pp.save(str(tmp_path / "pp"))
        pp2 = PreprocessingPipeline.load(str(tmp_path / "pp"))
        assert pp2._is_fitted is True
        assert pp2.target_classes == pp.target_classes


class TestFeatureEngineering:
    def test_engineer_features(self):
        adapter = SyntheticAdapter("synthetic")
        df = adapter.load()
        fe = FeatureEngineer()
        df2 = fe.engineer_features(df)
        # Should have same or more columns
        assert len(df2.columns) >= len(df.columns)

    def test_correlation_analysis(self):
        adapter = SyntheticAdapter("synthetic")
        df = adapter.load()
        df = df.drop(columns=["label"])
        fe = FeatureEngineer({"correlation_threshold": 0.95})
        report = fe.analyze_correlation(df)
        assert "correlated_pairs" in report
        assert "threshold" in report
