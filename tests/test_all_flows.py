"""
Comprehensive test suite for AutoML – tests both Supervised and Unsupervised flows.
Run: python tests/test_all_flows.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Ensure the project root is on sys.path so `from src.superml_forge...` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════

PASSED = 0
FAILED = 0


def run_test(name: str, func):
    global PASSED, FAILED
    try:
        func()
        print(f"  [PASS] {name}")
        PASSED += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAILED += 1


# ═══════════════════════════════════════════════════════════
# SAMPLE DATA
# ═══════════════════════════════════════════════════════════

np.random.seed(42)

# Classification dataset
clf_df = pd.DataFrame({
    "feature_a": np.random.randn(100),
    "feature_b": np.random.randn(100),
    "feature_c": np.random.choice(["cat", "dog", "bird"], 100),
    "target": np.random.choice(["yes", "no"], 100),
})

# Regression dataset
reg_df = pd.DataFrame({
    "size": np.random.uniform(500, 3000, 100),
    "rooms": np.random.randint(1, 6, 100),
    "age": np.random.uniform(1, 50, 100),
    "price": np.random.uniform(100000, 500000, 100),
})

# Unsupervised dataset (no target)
unsup_df = pd.DataFrame({
    "x1": np.random.randn(100),
    "x2": np.random.randn(100),
    "x3": np.random.randn(100),
    "x4": np.random.randn(100),
})


# ═══════════════════════════════════════════════════════════
# SUPERVISED TESTS
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("SUPERVISED LEARNING TESTS")
print("=" * 60)


def test_ingestion_imports():
    from src.superml_forge.ingestion import load_data, detect_extension, list_excel_sheets
    from src.superml_forge.ingestion import SUPPORTED_EXTENSIONS
    assert len(SUPPORTED_EXTENSIONS) >= 7

run_test("Ingestion imports", test_ingestion_imports)


def test_ingestion_detect_extension():
    from src.superml_forge.ingestion import detect_extension
    assert detect_extension("data.csv") == ".csv"
    assert detect_extension("report.xlsx") == ".xlsx"
    assert detect_extension("file.json") == ".json"

run_test("Ingestion detect_extension", test_ingestion_detect_extension)


def test_profiling():
    from src.superml_forge.profiling import DatasetInfo, get_dataset_info
    info = get_dataset_info(clf_df)
    assert isinstance(info, DatasetInfo)
    assert info.shape == (100, 4)
    assert len(info.head) == 5

run_test("Profiling DatasetInfo", test_profiling)


def test_task_detection_classification():
    from src.superml_forge.task_detection import detect_problem_type
    result = detect_problem_type(clf_df["target"])
    assert result == "classification"

run_test("Task detection (classification)", test_task_detection_classification)


def test_task_detection_regression():
    from src.superml_forge.task_detection import detect_problem_type
    result = detect_problem_type(reg_df["price"])
    assert result == "regression"

run_test("Task detection (regression)", test_task_detection_regression)


def test_preprocessing_pipeline():
    from src.superml_forge.preprocessing import build_preprocessor, build_full_pipeline
    from sklearn.linear_model import LogisticRegression
    X = clf_df.drop(columns=["target"])
    preprocessor, num_feats, cat_feats = build_preprocessor(X)
    assert len(num_feats) >= 2
    assert len(cat_feats) >= 1
    # Test full pipeline
    pipeline = build_full_pipeline(X, LogisticRegression())
    pipeline.fit(X, clf_df["target"])
    preds = pipeline.predict(X)
    assert len(preds) == 100

run_test("Preprocessing pipeline", test_preprocessing_pipeline)


def test_splitting():
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(clf_df, "target", "classification")
    assert len(X_train) == 80
    assert len(X_test) == 20

run_test("Train/test splitting", test_splitting)


def test_splitting_regression():
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(reg_df, "price", "regression")
    assert len(X_train) + len(X_test) == 100

run_test("Train/test splitting (regression)", test_splitting_regression)


def test_model_registry():
    from src.superml_forge.models import get_classification_models, get_regression_models
    clf_models = get_classification_models()
    reg_models = get_regression_models()
    assert len(clf_models) == 4
    assert len(reg_models) == 4
    assert "Logistic Regression" in clf_models
    assert "Random Forest" in clf_models
    assert "Linear Regression" in reg_models

run_test("Model registry", test_model_registry)


def test_tuning_param_grids():
    from src.superml_forge.tuning import get_classification_param_grids, get_regression_param_grids
    clf_grids = get_classification_param_grids()
    reg_grids = get_regression_param_grids()
    assert "Logistic Regression" in clf_grids
    assert "Random Forest Regressor" in reg_grids

run_test("Tuning param grids", test_tuning_param_grids)


def test_training_classification():
    from src.superml_forge.training import train_and_tune_models, ModelResult
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(clf_df, "target", "classification")
    best, all_results, comparison = train_and_tune_models(
        X_train, y_train, "classification", n_iter=2, cv_folds=2
    )
    assert isinstance(best, ModelResult)
    assert best.name in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
    assert len(all_results) == 4
    assert len(comparison) == 4
    # Test prediction
    preds = best.best_estimator.predict(X_test)
    assert len(preds) == len(X_test)

run_test("Training (classification, all models)", test_training_classification)


def test_training_regression():
    from src.superml_forge.training import train_and_tune_models, ModelResult
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(reg_df, "price", "regression")
    best, all_results, comparison = train_and_tune_models(
        X_train, y_train, "regression", n_iter=2, cv_folds=2
    )
    assert isinstance(best, ModelResult)
    assert len(all_results) == 4
    preds = best.best_estimator.predict(X_test)
    assert len(preds) == len(X_test)

run_test("Training (regression, all models)", test_training_regression)


def test_training_manual_mode():
    from src.superml_forge.training import train_and_tune_models
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(clf_df, "target", "classification")
    best, all_results, comparison = train_and_tune_models(
        X_train, y_train, "classification",
        selected_model_name="Random Forest",
        manual_params={"n_estimators": 50, "max_depth": 5},
        n_iter=2, cv_folds=2,
    )
    assert best.name == "Random Forest"
    assert len(all_results) == 1

run_test("Training (manual mode)", test_training_manual_mode)


def test_evaluation_classification():
    from src.superml_forge.evaluation import evaluate_classification
    from src.superml_forge.training import train_and_tune_models
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(clf_df, "target", "classification")
    best, _, _ = train_and_tune_models(X_train, y_train, "classification", n_iter=2, cv_folds=2)
    metrics, cm, classes = evaluate_classification(best.best_estimator, X_test, y_test)
    assert "Accuracy" in metrics
    assert "F1-score (weighted)" in metrics
    assert cm.shape[0] > 0

run_test("Evaluation (classification)", test_evaluation_classification)


def test_evaluation_regression():
    from src.superml_forge.evaluation import evaluate_regression
    from src.superml_forge.training import train_and_tune_models
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(reg_df, "price", "regression")
    best, _, _ = train_and_tune_models(X_train, y_train, "regression", n_iter=2, cv_folds=2)
    metrics = evaluate_regression(best.best_estimator, X_test, y_test)
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "R^2" in metrics

run_test("Evaluation (regression)", test_evaluation_regression)


def test_prediction():
    from src.superml_forge.prediction import predict_with_input
    from src.superml_forge.training import train_and_tune_models
    from src.superml_forge.splitting import train_test_split_data
    X_train, X_test, y_train, y_test = train_test_split_data(clf_df, "target", "classification")
    best, _, _ = train_and_tune_models(X_train, y_train, "classification", n_iter=2, cv_folds=2)
    sample = X_test.head(1)
    preds = predict_with_input(best.best_estimator, sample, X_train.columns.tolist())
    assert len(preds) == 1

run_test("Prediction", test_prediction)


def test_validation():
    from src.superml_forge.validation import sanitize_dataframe, get_missing_value_summary
    cleaned = sanitize_dataframe(clf_df)
    assert cleaned.shape == clf_df.shape
    summary = get_missing_value_summary(cleaned)
    assert isinstance(summary, pd.DataFrame)

run_test("Validation (sanitize + missing)", test_validation)


def test_selection():
    from src.superml_forge.selection.best_model import select_best
    from src.superml_forge.selection.ranking import rank_models
    from src.superml_forge.training import ModelResult
    from sklearn.linear_model import LogisticRegression
    results = [
        ModelResult("A", LogisticRegression(), 0.9, {}),
        ModelResult("B", LogisticRegression(), 0.95, {}),
        ModelResult("C", LogisticRegression(), 0.85, {}),
    ]
    best = select_best(results, "classification")
    assert best.name == "B"
    ranked = rank_models(results, "classification")
    assert ranked[0].name == "B"

run_test("Selection (best model + ranking)", test_selection)


# ═══════════════════════════════════════════════════════════
# UNSUPERVISED TESTS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("UNSUPERVISED LEARNING TESTS")
print("=" * 60)


def test_unsupervised_imports():
    from src.superml_forge.unsupervised import run_unsupervised_workflow
    from src.superml_forge.unsupervised.clustering import get_clustering_models
    from src.superml_forge.unsupervised.dimensionality_reduction import get_reduction_models
    from src.superml_forge.unsupervised.anomaly_detection import get_anomaly_models
    from src.superml_forge.unsupervised.evaluation import (
        compute_clustering_metrics, compute_reduction_metrics, compute_anomaly_metrics
    )

run_test("Unsupervised imports", test_unsupervised_imports)


def test_preprocessor():
    from src.superml_forge.unsupervised.preprocessor import preprocess_for_unsupervised
    X, names, df_clean = preprocess_for_unsupervised(unsup_df)
    assert X.shape == (100, 4)
    assert len(names) == 4

run_test("Unsupervised preprocessor", test_preprocessor)


def test_preprocessor_with_columns():
    from src.superml_forge.unsupervised.preprocessor import preprocess_for_unsupervised
    X, names, _ = preprocess_for_unsupervised(unsup_df, columns=["x1", "x2"])
    assert X.shape == (100, 2)
    assert names == ["x1", "x2"]

run_test("Unsupervised preprocessor (column selection)", test_preprocessor_with_columns)


def test_clustering_registry():
    from src.superml_forge.unsupervised.clustering.registry import get_clustering_models, get_clustering_param_options
    models = get_clustering_models()
    assert len(models) == 6
    assert "K-Means" in models
    assert "DBSCAN" in models
    assert "Gaussian Mixture" in models
    opts = get_clustering_param_options()
    assert "K-Means" in opts

run_test("Clustering registry (6 algorithms)", test_clustering_registry)


def test_clustering_all_models():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(unsup_df, "clustering")
    assert len(result["results"]) == 6
    assert result["best"] is not None
    assert result["best"].n_clusters >= 1
    assert "comparison" in result

run_test("Clustering (all 6 models)", test_clustering_all_models)


def test_clustering_manual_kmeans():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(
        unsup_df, "clustering", model_name="K-Means", params={"n_clusters": 5}
    )
    assert len(result["results"]) == 1
    assert result["best"].name == "K-Means"
    assert result["best"].n_clusters == 5

run_test("Clustering (manual K-Means, k=5)", test_clustering_manual_kmeans)


def test_clustering_manual_dbscan():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(
        unsup_df, "clustering", model_name="DBSCAN", params={"eps": 1.0, "min_samples": 3}
    )
    assert result["best"].name == "DBSCAN"

run_test("Clustering (manual DBSCAN)", test_clustering_manual_dbscan)


def test_clustering_metrics():
    from src.superml_forge.unsupervised.evaluation import compute_clustering_metrics
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(unsup_df.values)
    metrics = compute_clustering_metrics(unsup_df.values, labels)
    assert "silhouette_score" in metrics
    assert "calinski_harabasz" in metrics
    assert "davies_bouldin" in metrics

run_test("Clustering metrics (silhouette, calinski, davies)", test_clustering_metrics)


def test_reduction_registry():
    from src.superml_forge.unsupervised.dimensionality_reduction.registry import get_reduction_models
    models = get_reduction_models()
    assert len(models) == 3
    assert "PCA" in models
    assert "t-SNE" in models

run_test("Reduction registry (3 algorithms)", test_reduction_registry)


def test_reduction_all_models():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(unsup_df, "dimensionality_reduction")
    assert len(result["results"]) == 3
    assert result["best"] is not None
    assert result["best"].transformed.shape[0] == 100
    assert result["best"].transformed.shape[1] == 2

run_test("Reduction (all 3 models)", test_reduction_all_models)


def test_reduction_pca_variance():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(
        unsup_df, "dimensionality_reduction", model_name="PCA", params={"n_components": 3}
    )
    assert result["best"].name == "PCA"
    assert "total_explained_variance" in result["best"].metrics
    assert result["best"].metrics["total_explained_variance"] > 0

run_test("Reduction (PCA explained variance)", test_reduction_pca_variance)


def test_anomaly_registry():
    from src.superml_forge.unsupervised.anomaly_detection.registry import get_anomaly_models
    models = get_anomaly_models()
    assert len(models) == 3
    assert "Isolation Forest" in models
    assert "Local Outlier Factor" in models
    assert "One-Class SVM" in models

run_test("Anomaly registry (3 algorithms)", test_anomaly_registry)


def test_anomaly_all_models():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(unsup_df, "anomaly_detection")
    assert len(result["results"]) == 3
    assert result["best"] is not None
    assert result["best"].n_anomalies >= 0
    for r in result["results"]:
        if r.labels.size > 0:
            assert set(r.labels).issubset({-1, 1})

run_test("Anomaly detection (all 3 models)", test_anomaly_all_models)


def test_anomaly_manual_isolation_forest():
    from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
    result = run_unsupervised_workflow(
        unsup_df, "anomaly_detection",
        model_name="Isolation Forest",
        params={"n_estimators": 200, "contamination": 0.05}
    )
    assert result["best"].name == "Isolation Forest"
    assert result["best"].anomaly_pct <= 15  # roughly around 5%

run_test("Anomaly (manual Isolation Forest)", test_anomaly_manual_isolation_forest)


def test_anomaly_metrics():
    from src.superml_forge.unsupervised.evaluation import compute_anomaly_metrics
    labels = np.array([1, 1, 1, -1, 1, 1, -1, 1, 1, 1])
    scores = np.random.randn(10)
    metrics = compute_anomaly_metrics(labels, scores)
    assert metrics["n_anomalies"] == 2
    assert metrics["anomaly_pct"] == 20.0
    assert metrics["n_normal"] == 8

run_test("Anomaly metrics", test_anomaly_metrics)


# ═══════════════════════════════════════════════════════════
# SHARED MODULE TESTS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SHARED MODULE TESTS")
print("=" * 60)


def test_config():
    from src.superml_forge.config import Settings
    s = Settings()
    assert s.app_name == "AutoML"

run_test("Config settings", test_config)


def test_utils_logger():
    from src.superml_forge.utils.logger import get_logger
    logger = get_logger("test")
    assert logger is not None

run_test("Utils logger", test_utils_logger)


def test_utils_exceptions():
    from src.superml_forge.utils.exceptions import (
        SuperMLError, DataIngestionError, ValidationError,
        PreprocessingError, TrainingError, PredictionError
    )
    assert issubclass(DataIngestionError, SuperMLError)
    assert issubclass(TrainingError, SuperMLError)

run_test("Utils exceptions", test_utils_exceptions)


def test_domain_objects():
    from src.superml_forge.domain.dataset import Dataset
    from src.superml_forge.domain.task import Task
    from src.superml_forge.domain.features import FeatureSet
    ds = Dataset(dataframe=clf_df, name="test")
    assert ds.shape == (100, 4)
    t = Task(problem_type="classification", target_column="target")
    assert t.problem_type == "classification"
    fs = FeatureSet(numeric=["a", "b"], categorical=["c"])
    assert len(fs.all_features) == 3

run_test("Domain objects", test_domain_objects)


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = PASSED + FAILED
print(f"RESULTS: {PASSED}/{total} passed, {FAILED}/{total} failed")
if FAILED == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {FAILED} test(s) failed.")
print("=" * 60)

sys.exit(0 if FAILED == 0 else 1)
