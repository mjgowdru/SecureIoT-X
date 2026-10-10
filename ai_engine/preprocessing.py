"""
SecureIoT-X AI Engine — Preprocessing Pipeline.

CRITICAL: Fit transformations ONLY on training data to prevent data leakage.
Then transform train, validation, and test sets separately.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

logger = logging.getLogger("secureiot.ai.preprocessing")


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline with leakage prevention.

    Usage:
        pp = PreprocessingPipeline(config)
        X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(X, y)
        pp.save("models/preprocessing/")

        # Later
        pp2 = PreprocessingPipeline.load("models/preprocessing/")
        X_new = pp2.transform(X_new_raw)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.test_size: float = cfg.get("test_size", 0.2)
        self.val_size: float = cfg.get("validation_size", 0.1)
        self.random_seed: int = cfg.get("random_seed", 42)
        self.scaling_method: str = cfg.get("scaling_method", "standard")
        self.handle_inf: bool = cfg.get("handle_inf", True)
        self.remove_duplicates: bool = cfg.get("remove_duplicates", True)
        self.max_missing_ratio: float = cfg.get("max_missing_ratio", 0.5)

        # Fitted transformers (set during fit)
        self._scaler: Optional[Any] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._target_encoder: Optional[LabelEncoder] = None
        self._feature_columns: List[str] = []
        self._numerical_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._dropped_columns: List[str] = []
        self._is_fitted = False

    # ── Main API ───────────────────────────────────────────────

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline: clean → split → fit on train → transform all splits.
        Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Clean data
        X_clean, y_clean = self._clean(X, y)

        # Step 2: Split BEFORE fitting — prevents data leakage
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_clean, y_clean,
            test_size=self.test_size + self.val_size,
            random_state=self.random_seed,
            stratify=y_clean if self._can_stratify(y_clean) else None,
        )

        if self.val_size > 0:
            val_ratio = self.val_size / (self.test_size + self.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=1 - val_ratio,
                random_state=self.random_seed,
                stratify=y_temp if self._can_stratify(y_temp) else None,
            )
        else:
            X_val, y_val = pd.DataFrame(), pd.Series(dtype=y_clean.dtype)
            X_test, y_test = X_temp, y_temp

        logger.info("Split sizes — train: %d, val: %d, test: %d",
                     len(X_train), len(X_val), len(X_test))

        # Step 3: Identify column types
        self._numerical_columns = list(X_train.select_dtypes(include=[np.number]).columns)
        self._categorical_columns = list(X_train.select_dtypes(include=["object", "category"]).columns)
        self._feature_columns = list(X_train.columns)

        # Step 4: Encode categoricals — FIT on train only
        X_train = self._fit_encode_categoricals(X_train)
        X_val = self._transform_encode_categoricals(X_val)
        X_test = self._transform_encode_categoricals(X_test)

        # Step 5: Scale numericals — FIT on train only
        X_train = self._fit_scale(X_train)
        X_val = self._transform_scale(X_val)
        X_test = self._transform_scale(X_test)

        # Step 6: Encode target labels
        self._target_encoder = LabelEncoder()
        y_train_enc = self._target_encoder.fit_transform(y_train)
        y_val_enc = self._target_encoder.transform(y_val) if len(y_val) > 0 else np.array([])
        y_test_enc = self._target_encoder.transform(y_test)

        self._is_fitted = True
        logger.info("Preprocessing complete. Features: %d", X_train.shape[1])

        return (
            X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
            X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
            y_train_enc, y_val_enc, y_test_enc,
        )

    def transform(self, X: Any) -> np.ndarray:
        """Transform new data using fitted pipeline (for inference)."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first or load().")

        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        # Align columns to fitted feature columns
        if self._feature_columns:
            for col in self._feature_columns:
                if col not in X.columns:
                    if col in self._numerical_columns:
                        X[col] = 0.0
                    elif col in self._categorical_columns and col in self._label_encoders:
                        X[col] = self._label_encoders[col].classes_[0]
                    else:
                        X[col] = 0.0
            # Retain only the exact feature columns in fitted order
            X = X[self._feature_columns]

            # Enforce numeric types on numerical columns
            for col in self._numerical_columns:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

        X = self._handle_inf_values(X)
        X = self._handle_missing(X)
        X = self._transform_encode_categoricals(X)
        X = self._transform_scale(X)
        return X.values if isinstance(X, pd.DataFrame) else X

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original strings."""
        if self._target_encoder is None:
            raise RuntimeError("Target encoder not fitted")
        return self._target_encoder.inverse_transform(y)

    @property
    def target_classes(self) -> List[str]:
        if self._target_encoder is None:
            return []
        return list(self._target_encoder.classes_)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_columns.copy()

    # ── Cleaning ───────────────────────────────────────────────

    def _clean(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean the raw data before splitting."""
        X = X.copy()
        y = y.copy()
        initial_rows = len(X)

        # Remove duplicates
        if self.remove_duplicates:
            combined = X.copy()
            combined["__target__"] = y
            before = len(combined)
            combined = combined.drop_duplicates()
            after = len(combined)
            if before != after:
                logger.info("Removed %d duplicate rows", before - after)
            y = combined.pop("__target__")
            X = combined

        # Drop columns with too many missing values
        missing_ratio = X.isnull().mean()
        high_missing = missing_ratio[missing_ratio > self.max_missing_ratio].index.tolist()
        if high_missing:
            logger.info("Dropping %d columns with >%.0f%% missing: %s",
                        len(high_missing), self.max_missing_ratio * 100, high_missing)
            self._dropped_columns.extend(high_missing)
            X = X.drop(columns=high_missing)

        # Handle infinities
        X = self._handle_inf_values(X)

        # Handle remaining missing values
        X = self._handle_missing(X)

        # Remove rows where target is missing
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        logger.info("Cleaning: %d → %d rows", initial_rows, len(X))
        return X, y

    def _handle_inf_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN, then handle as missing."""
        if self.handle_inf:
            num_cols = [c for c in self._numerical_columns if c in X.columns] if self._numerical_columns else list(X.select_dtypes(include=[np.number]).columns)
            if num_cols:
                inf_count = np.isinf(X[num_cols]).sum().sum()
                if inf_count > 0:
                    logger.info("Replacing %d infinite values with NaN", inf_count)
                    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
        return X

    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values — median for numerical, mode for categorical."""
        num_cols = [c for c in self._numerical_columns if c in X.columns] if self._numerical_columns else list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in self._categorical_columns if c in X.columns] if self._categorical_columns else list(X.select_dtypes(include=["object", "category"]).columns)

        if len(num_cols) > 0:
            medians = X[num_cols].median().fillna(0.0)
            X[num_cols] = X[num_cols].fillna(medians).fillna(0.0)
        if len(cat_cols) > 0:
            for col in cat_cols:
                mode_val = X[col].mode()
                default_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "UNKNOWN"
                X[col] = X[col].fillna(default_val)
        return X

    # ── Encoding ───────────────────────────────────────────────

    def _fit_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit label encoders on training data categorical columns."""
        self._label_encoders = {}
        for col in self._categorical_columns:
            if col not in X.columns:
                continue
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le
        return X

    def _transform_encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders."""
        if len(X) == 0:
            return X
        for col, le in self._label_encoders.items():
            if col not in X.columns:
                continue
            # Handle unseen categories gracefully
            X[col] = X[col].astype(str)
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])
        return X

    # ── Scaling ────────────────────────────────────────────────

    def _fit_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data numerical columns."""
        if self.scaling_method == "standard":
            self._scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            logger.warning("Unknown scaling method '%s', using StandardScaler", self.scaling_method)
            self._scaler = StandardScaler()

        num_cols = [c for c in self._numerical_columns if c in X.columns]
        if num_cols:
            X[num_cols] = self._scaler.fit_transform(X[num_cols])
        return X

    def _transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if self._scaler is None or len(X) == 0:
            return X
        num_cols = [c for c in self._numerical_columns if c in X.columns]
        if num_cols:
            X[num_cols] = self._scaler.transform(X[num_cols])
        return X

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _can_stratify(y: pd.Series) -> bool:
        """Check if stratification is possible (each class has ≥2 samples)."""
        if len(y) == 0:
            return False
        counts = y.value_counts()
        return counts.min() >= 2

    # ── Serialisation ──────────────────────────────────────────

    def save(self, dirpath: str) -> None:
        """Save the fitted pipeline to disk."""
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "scaler": self._scaler,
            "label_encoders": self._label_encoders,
            "target_encoder": self._target_encoder,
            "feature_columns": self._feature_columns,
            "numerical_columns": self._numerical_columns,
            "categorical_columns": self._categorical_columns,
            "dropped_columns": self._dropped_columns,
            "scaling_method": self.scaling_method,
            "handle_inf": self.handle_inf,
        }
        joblib.dump(state, path / "preprocessing_pipeline.joblib")
        logger.info("Preprocessing pipeline saved to %s", path)

    @classmethod
    def load(cls, dirpath: str) -> "PreprocessingPipeline":
        """Load a fitted pipeline from disk."""
        path = Path(dirpath)
        state = joblib.load(path / "preprocessing_pipeline.joblib")

        pp = cls()
        pp._scaler = state["scaler"]
        pp._label_encoders = state["label_encoders"]
        pp._target_encoder = state["target_encoder"]
        pp._feature_columns = state["feature_columns"]
        pp._numerical_columns = state["numerical_columns"]
        pp._categorical_columns = state["categorical_columns"]
        pp._dropped_columns = state["dropped_columns"]
        pp.scaling_method = state["scaling_method"]
        pp.handle_inf = state["handle_inf"]
        pp._is_fitted = True
        logger.info("Preprocessing pipeline loaded from %s", path)
        return pp
