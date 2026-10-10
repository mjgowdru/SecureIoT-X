"""
SecureIoT-X AI Engine — Main Training Pipeline Entry Point.

Usage: python -m ai_engine

Runs the full pipeline:
1. Load dataset
2. Validate dataset
3. Preprocess (with leakage prevention)
4. Engineer features
5. Train all models
6. Evaluate all models
7. Train anomaly detector
8. Save everything
9. Generate comparison report
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ai_engine.data_loader import get_adapter
from ai_engine.preprocessing import PreprocessingPipeline
from ai_engine.feature_engineering import FeatureEngineer
from ai_engine.train import train_all_models
from ai_engine.evaluate import evaluate_all_models, select_best_model, generate_comparison_report
from ai_engine.anomaly_detection import AnomalyDetector
from ai_engine.model_registry import ModelRegistry
from ai_engine.explainability import Explainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("secureiot.ai")


def _load_ml_config() -> dict:
    """Load ML configuration from YAML."""
    import yaml
    config_path = _root / "config" / "ml.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    """Run the complete AI training pipeline."""
    print("=" * 70)
    print("  SecureIoT-X — AI/ML Training Pipeline")
    print("=" * 70)

    ml_config = _load_ml_config()
    pipeline_cfg = ml_config.get("pipeline", {})
    preprocess_cfg = ml_config.get("preprocessing", {})
    feat_cfg = ml_config.get("feature_selection", {})
    model_configs = ml_config.get("models", [])
    anomaly_cfg = ml_config.get("anomaly_detection", {})
    storage_cfg = ml_config.get("model_storage", {})

    random_seed = pipeline_cfg.get("random_seed", 42)
    models_dir = storage_cfg.get("base_dir", "models")
    preprocessing_dir = storage_cfg.get("preprocessing_dir", "models/preprocessing")
    reports_dir = storage_cfg.get("reports_dir", "reports")

    # ── Step 1: Load Dataset ───────────────────────────────────
    print("\n[1/9] Loading dataset...")
    dataset_name = ml_config.get("datasets", {}).get("primary", "Synthetic")
    datasets_list = ml_config.get("datasets", {}).get("supported", [])

    # Find dataset path
    dataset_path = ""
    for ds in datasets_list:
        if ds["name"] == dataset_name:
            dataset_path = str(_root / ds["path"])
            break

    # Check if primary dataset exists, fall back to Synthetic
    if dataset_path and not Path(dataset_path).exists():
        print(f"  [!] Dataset '{dataset_name}' not found at {dataset_path}")
        print("  -> Falling back to Synthetic dataset for development")
        dataset_name = "Synthetic"

    adapter = get_adapter(dataset_name, dataset_path)
    df = adapter.load()

    # ── Step 2: Validate ───────────────────────────────────────
    print("\n[2/9] Validating dataset...")
    target_col = adapter.identify_target()
    report = adapter.validate()
    adapter.print_report(report)

    X, y = adapter.get_features_and_target()

    # ── Step 3: Feature Engineering ────────────────────────────
    print("\n[3/9] Engineering features...")
    fe = FeatureEngineer(feat_cfg)
    X = fe.engineer_features(X)

    # Correlation analysis
    corr_report = fe.analyze_correlation(X)
    if corr_report["dropped"]:
        X = fe.remove_correlated(X, corr_report["dropped"])

    # ── Step 4: Preprocess ─────────────────────────────────────
    print("\n[4/9] Preprocessing (with data leakage prevention)...")
    pp_config = {
        "test_size": pipeline_cfg.get("test_size", 0.2),
        "validation_size": pipeline_cfg.get("validation_size", 0.1),
        "random_seed": random_seed,
        **preprocess_cfg,
    }
    pp = PreprocessingPipeline(pp_config)
    X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(X, y)

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {pp.target_classes}")

    # ── Step 5: Train Models ───────────────────────────────────
    print("\n[5/9] Training models...")
    if not model_configs:
        # Default models if config missing
        model_configs = [
            {"name": "LogisticRegression", "enabled": True, "params": {"max_iter": 1000, "random_state": random_seed}},
            {"name": "DecisionTree", "enabled": True, "params": {"max_depth": 20, "random_state": random_seed}},
            {"name": "RandomForest", "enabled": True, "params": {"n_estimators": 100, "random_state": random_seed, "n_jobs": -1}},
            {"name": "SVM", "enabled": True, "params": {"kernel": "rbf", "random_state": random_seed, "max_iter": 5000}},
            {"name": "GradientBoosting", "enabled": True, "params": {"n_estimators": 100, "random_state": random_seed}},
        ]

    train_results = train_all_models(
        X_train, y_train,
        model_configs=model_configs,
        save_dir=models_dir,
        dataset_name=dataset_name,
        feature_names=pp.feature_names,
        random_seed=random_seed,
    )

    # ── Step 6: Evaluate Models ────────────────────────────────
    print("\n[6/9] Evaluating models...")
    evaluated = evaluate_all_models(
        train_results, X_test, y_test,
        class_names=pp.target_classes,
    )

    # ── Step 7: Select Best & Generate Report ──────────────────
    print("\n[7/9] Generating comparison report...")
    best = select_best_model(evaluated)
    report = generate_comparison_report(evaluated, save_path=reports_dir)

    # Feature importance from best model
    explainer = Explainer(pp.feature_names)
    importance_report = explainer.explain_model(best["model"])

    # ── Step 8: Train Anomaly Detector ─────────────────────────
    print("\n[8/9] Training anomaly detector...")
    ad = AnomalyDetector(anomaly_cfg)
    ad.fit(X_train)
    ad.save(str(Path(models_dir) / "anomaly"))

    # ── Step 9: Save Everything ────────────────────────────────
    print("\n[9/9] Saving pipeline & registry...")
    pp.save(preprocessing_dir)

    # Register models
    registry = ModelRegistry(str(Path(models_dir) / "registry.json"))
    for r in evaluated:
        metrics = r.get("metrics", {})
        registry.register(
            model_name=r["name"],
            version=r.get("version", "v1"),
            dataset=dataset_name,
            features=pp.feature_names,
            metrics=metrics,
            model_path=r.get("model_path", ""),
            preprocessing_path=preprocessing_dir,
            random_seed=random_seed,
        )

    # Validate and promote best model
    registry.validate_model(best["name"], best.get("version", "v1"))
    registry.set_production(best["name"], best.get("version", "v1"))

    # Save full report
    full_report = {
        "dataset": dataset_name,
        "random_seed": random_seed,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "classes": pp.target_classes,
        "comparison": report["comparison"],
        "best_model": best["name"],
        "feature_importance": importance_report,
        "anomaly_detector": anomaly_cfg.get("method", "isolation_forest"),
        "note": "These are OUR implementation results, not reference paper results.",
    }
    full_path = Path(reports_dir) / "full_training_report.json"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  [OK] TRAINING PIPELINE COMPLETE")
    print(f"  [*] Best model: {best['name']}")
    print(f"  [*] F1 Score:   {best['metrics']['f1_weighted']:.4f}")
    print(f"  [*] Recall:     {best['metrics']['recall_weighted']:.4f}")
    print(f"  Reports:      {reports_dir}/")
    print(f"  Models:       {models_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
