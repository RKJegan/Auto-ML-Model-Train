from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

from src.superml_forge.evaluation import evaluate_regression
from src.superml_forge.ingestion import detect_extension, list_excel_sheets, load_data
from src.superml_forge.models import get_classification_models, get_regression_models
from src.superml_forge.training import (
    ModelResult,
    train_and_tune_models,
)
from src.superml_forge.prediction import predict_with_input
from src.superml_forge.profiling import DatasetInfo, get_dataset_info
from src.superml_forge.task_detection import detect_problem_type
from src.superml_forge.splitting import train_test_split_data
from src.superml_forge.unsupervised.workflow import run_unsupervised_workflow
from src.superml_forge.unsupervised.clustering.registry import (
    get_clustering_models,
    get_clustering_param_options,
)
# pyrefly: ignore [missing-import]
from src.superml_forge.unsupervised.dimensionality_reduction.registry import (
    get_reduction_models,
    get_reduction_param_options,
)
from src.superml_forge.unsupervised.anomaly_detection.registry import (
    get_anomaly_models,
    get_anomaly_param_options,
)


st.set_page_config(
    page_title="AutoML",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════

for key, default in [
    ("trained_model", None),
    ("feature_columns", None),
    ("problem_type", None),
    ("target_column", None),
    ("numeric_columns", None),
    ("loaded_df", None),
    ("unsupervised_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════

@st.cache_data
def cached_load_data(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    return load_data(uploaded_file, sheet_name=sheet_name)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)
    return cleaned


def show_missing_value_summary(df: pd.DataFrame) -> None:
    null_counts = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": null_counts,
            "missing_pct": ((null_counts / len(df)) * 100).round(2),
        }
    )
    summary = summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
    st.subheader("Missing Value Summary")
    if summary.empty:
        st.success("No missing values found in this dataset.")
    else:
        st.dataframe(summary)


def display_dataset_info(info: DatasetInfo) -> None:
    st.subheader("Dataset Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape** (rows, columns):", info.shape)
        st.write("**Column types:**")
        st.dataframe(info.dtypes.astype(str).to_frame("dtype"))

    with col2:
        st.write("**Head:**")
        st.dataframe(info.head)


# ═══════════════════════════════════════════════════════════
# SUPERVISED LEARNING UI FUNCTIONS
# ═══════════════════════════════════════════════════════════

def build_prediction_input_form(feature_columns, numeric_columns, reference_df: pd.DataFrame) -> Tuple[bool, Dict]:
    with st.form("prediction_form"):
        st.caption("Empty or whitespace-only values are treated as missing and handled safely.")
        user_input: Dict = {}
        cols = st.columns(2)

        for i, col_name in enumerate(feature_columns):
            container = cols[i % 2]
            with container:
                if col_name in numeric_columns:
                    raw_value = st.text_input(f"{col_name} (numeric)", value="")
                    user_input[col_name] = raw_value
                else:
                    series = reference_df[col_name] if col_name in reference_df.columns else pd.Series(dtype="object")
                    unique_values = (
                        series.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
                    )
                    unique_values = unique_values[:50]
                    if 0 < len(unique_values) <= 50:
                        choice = st.selectbox(f"{col_name}", options=[""] + unique_values)
                        user_input[col_name] = choice
                    else:
                        user_input[col_name] = st.text_input(f"{col_name}", value="")

        submitted = st.form_submit_button("Predict")
    return submitted, user_input


def input_dict_to_dataframe(user_input: Dict, feature_columns, numeric_columns) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        value = user_input.get(col, np.nan)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                value = np.nan
        if col in numeric_columns:
            row[col] = pd.to_numeric(value, errors="coerce")
        else:
            row[col] = value
    return pd.DataFrame([row], columns=feature_columns)


def select_supervised_mode() -> str:
    return st.sidebar.radio(
        "Select mode",
        options=["Automatic Model Selection", "Manual Model Selection"],
        index=0,
    )


def get_manual_model_choice(problem_type: str) -> Optional[str]:
    """Let the user pick a specific model name in manual mode."""
    if problem_type == "classification":
        model_names = list(get_classification_models().keys())
    else:
        model_names = list(get_regression_models().keys())

    return st.sidebar.selectbox("Choose algorithm", options=model_names)


def get_manual_hyperparameters(problem_type: str, model_name: Optional[str]) -> Optional[Dict]:
    """
    Render hyperparameter controls in the sidebar for the selected model (manual mode)
    and return the chosen values as a dict that can be passed to the estimator.
    """
    if model_name is None:
        return None

    with st.sidebar.expander("Model Hyperparameters", expanded=True):
        if problem_type == "classification":
            if model_name == "Logistic Regression":
                C = st.number_input("C (inverse regularization strength)", 0.0001, 1000.0, 1.0, step=0.1)
                max_iter = st.number_input("max_iter", 100, 5000, 1000, step=100)
                penalty = st.selectbox("penalty", ["l2"])
                solver = st.selectbox("solver", ["lbfgs", "liblinear"])
                return {"C": C, "max_iter": max_iter, "penalty": penalty, "solver": solver}

            if model_name == "Decision Tree":
                criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }

            if model_name == "Random Forest":
                n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10)
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }

            if model_name == "SVM":
                C = st.number_input("C", 0.0001, 1000.0, 1.0, step=0.1)
                kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
                gamma = st.selectbox("gamma", ["scale", "auto"])
                return {"C": C, "kernel": kernel, "gamma": gamma}

        else:
            # Regression models
            if model_name == "Linear Regression":
                fit_intercept = st.checkbox("fit_intercept", value=True)
                return {"fit_intercept": fit_intercept}

            if model_name == "Ridge Regression":
                alpha = st.number_input("alpha", 0.0001, 1000.0, 1.0, step=0.1)
                fit_intercept = st.checkbox("fit_intercept", value=True)
                return {"alpha": alpha, "fit_intercept": fit_intercept}

            if model_name == "Lasso Regression":
                alpha = st.number_input("alpha", 0.0001, 1000.0, 1.0, step=0.1)
                max_iter = st.number_input("max_iter", 100, 5000, 1000, step=100)
                return {"alpha": alpha, "max_iter": max_iter}

            if model_name == "Random Forest Regressor":
                n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10)
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }

    return None


def display_model_results(
    best_result: ModelResult,
    comparison_df: pd.DataFrame,
    problem_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    target_column: str,
) -> None:
    st.subheader("Best Model Summary")
    st.write(f"**Best algorithm:** {best_result.name}")

    if problem_type == "classification":
        y_pred = getattr(best_result.best_estimator, "predict")(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        st.write("**Accuracy (test set):**")
        st.metric("Accuracy", f"{max(0.0, acc) * 100:.2f}%")
    else:
        metrics = evaluate_regression(best_result.best_estimator, X_test, y_test)
        st.write("**Evaluation metrics (test set):**")
        st.json({k: float(v) for k, v in metrics.items()})

    st.subheader("Best Hyperparameters")
    if best_result.best_params:
        st.json(best_result.best_params)
    else:
        st.write("No hyperparameters were tuned for this model.")

    # Model comparison table intentionally hidden in UI.

    # ─── Download trained model ───
    render_model_download(best_result.best_estimator, best_result.name, problem_type, target_column)


def render_model_download(
    model,
    model_name: str,
    problem_type: str,
    target_column: str,
) -> None:
    """
    Render a download button that lets the user save the trained model as a .pkl file.
    Uses joblib for serialization (handles numpy arrays and sklearn pipelines well).
    """
    import joblib

    st.subheader("Download Trained Model")
    st.info(
        f"Download the **{model_name}** model trained for "
        f"**{problem_type}** (target: `{target_column}`) as a `.pkl` file. "
        f"Load it later with `joblib.load('model.pkl')` to make predictions."
    )

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    safe_name = model_name.lower().replace(" ", "_")
    file_name = f"{safe_name}_{problem_type}_{target_column}.pkl"

    st.download_button(
        label=f"⬇️ Download {model_name} (.pkl)",
        data=buffer,
        file_name=file_name,
        mime="application/octet-stream",
        key="download_supervised_model",
    )


def render_prediction_section(df: pd.DataFrame) -> None:
    """
    Show prediction form only when a trained model exists in session_state.
    Uses feature_columns for correct column order; validates against current df when building inputs.
    """
    st.subheader("Prediction")
    if st.session_state["trained_model"] is None:
        st.warning("Please train the model first.")
        return

    model_result: ModelResult = st.session_state["trained_model"]
    feature_columns = st.session_state["feature_columns"]
    problem_type = st.session_state["problem_type"]
    numeric_columns = st.session_state["numeric_columns"] or []

    if not feature_columns:
        st.warning("No feature columns saved. Please run AutoML again.")
        return

    target_column = st.session_state.get("target_column")
    st.info(
        f"Prediction target: `{target_column}`. Provide feature values below; empty values are treated as missing."
    )

    submitted, user_input = build_prediction_input_form(feature_columns, numeric_columns, df)

    if submitted:
        try:
            input_df = input_dict_to_dataframe(user_input, feature_columns, numeric_columns)
            pred = predict_with_input(
                model=model_result.best_estimator,
                input_df=input_df,
                feature_columns=feature_columns,
            )[0]
            if problem_type == "classification":
                st.caption(f"Target prediction ({target_column})")
                st.success(f"**Predicted Class:** {pred}")
            else:
                st.caption(f"Target prediction ({target_column})")
                st.success(f"**Predicted Value:** {pred}")
        except ValueError as exc:
            st.error(f"Prediction validation error: {exc}")
        except Exception as e:
            st.error(
                "Prediction failed unexpectedly. Please verify the model was trained on the current dataset "
                "and try again."
            )


def run_supervised_flow(df: pd.DataFrame, uploaded_file) -> None:
    """Run the full supervised learning workflow."""
    # Target column selection
    target_column = st.selectbox("Select target column (what you want to predict)", df.columns)
    if not target_column:
        st.info("Select a target column to continue.")
        return

    target_series = df[target_column].replace(r"^\s*$", np.nan, regex=True).dropna()
    if target_series.empty:
        st.error("The selected target column has only null/blank values after cleaning.")
        return

    problem_type = detect_problem_type(target_series)

    if problem_type == "classification":
        st.success("Detected problem type: **Classification**")
    else:
        st.success("Detected problem type: **Regression**")

    mode = select_supervised_mode()
    selected_model_name: Optional[str] = None
    manual_params: Optional[Dict] = None
    if mode == "Manual Model Selection":
        selected_model_name = get_manual_model_choice(problem_type)
        manual_params = get_manual_hyperparameters(problem_type, selected_model_name)

    if st.button("Run AutoML"):
        with st.spinner("Training models. This may take a moment..."):
            try:
                train_df = df.copy()

                from src.superml_forge.feature_engineering.feature_selector import filter_important_features
                
                # Target column is excluded from features: X = df.drop(target), y = df[target]
                X_train, X_test, y_train, y_test = train_test_split_data(
                    train_df, target_column, problem_type
                )
                
                # Filter out irrelevant IDs and high cardinality features
                X_train = filter_important_features(X_train)
                X_test = X_test[X_train.columns]  # Keep only the selected features in test set

                best_result, all_results, comparison_df = train_and_tune_models(
                    X_train=X_train,
                    y_train=y_train,
                    problem_type=problem_type,
                    selected_model_name=selected_model_name,
                    manual_params=manual_params,
                )
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                return

        # Persist trained model and feature info so Predict button works after reruns
        st.session_state["trained_model"] = best_result
        st.session_state["feature_columns"] = X_train.columns.tolist()
        st.session_state["numeric_columns"] = X_train.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state["problem_type"] = problem_type
        st.session_state["target_column"] = target_column

        display_model_results(
            best_result=best_result,
            comparison_df=comparison_df,
            problem_type=problem_type,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            target_column=target_column,
        )

    # Prediction section: show only when a trained model exists; otherwise "Please train first"
    render_prediction_section(df)


# ═══════════════════════════════════════════════════════════
# UNSUPERVISED LEARNING UI FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_unsupervised_manual_params(task: str, model_name: str) -> Optional[Dict]:
    """Render hyperparameter controls for unsupervised models."""
    if task == "clustering":
        param_options = get_clustering_param_options()
    elif task == "dimensionality_reduction":
        param_options = get_reduction_param_options()
    elif task == "anomaly_detection":
        param_options = get_anomaly_param_options()
    else:
        return None

    if model_name not in param_options:
        return None

    options = param_options[model_name]
    params = {}

    with st.sidebar.expander("Model Hyperparameters", expanded=True):
        for param_name, config in options.items():
            if config["type"] == "int":
                params[param_name] = int(st.number_input(
                    param_name,
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=1,
                    key=f"unsup_{model_name}_{param_name}",
                ))
            elif config["type"] == "float":
                params[param_name] = float(st.number_input(
                    param_name,
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=0.01,
                    key=f"unsup_{model_name}_{param_name}",
                ))
            elif config["type"] == "select":
                params[param_name] = st.selectbox(
                    param_name,
                    options=config["options"],
                    index=config["options"].index(config["default"]),
                    key=f"unsup_{model_name}_{param_name}",
                )

    return params


def display_clustering_results(results: Dict) -> None:
    """Display clustering results with metrics and scatter plot."""
    best = results.get("best")
    comparison = results.get("comparison")
    all_results = results.get("results", [])

    if best is None:
        st.error("No clustering results available.")
        return

    st.subheader("Clustering Results")

    # Best model summary
    st.write(f"**Best Algorithm:** {best.name}")
    st.write(f"**Clusters Found:** {best.n_clusters}")

    # Metrics
    if best.metrics:
        metrics_display = {k: v for k, v in best.metrics.items() if isinstance(v, (int, float))}
        if metrics_display:
            metric_cols = st.columns(min(len(metrics_display), 4))
            for i, (k, v) in enumerate(metrics_display.items()):
                with metric_cols[i % len(metric_cols)]:
                    display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    st.metric(k.replace("_", " ").title(), display_val)

    # Comparison table
    if comparison is not None and not comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

    # Scatter plot of first two features colored by cluster label
    if best.labels is not None and len(best.labels) > 0:
        st.subheader("Cluster Visualization (First 2 Features)")
        try:
            from src.superml_forge.unsupervised.preprocessor import preprocess_for_unsupervised
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(
                results.get("_X_scaled", np.zeros((len(best.labels), 2)))[:, 0],
                results.get("_X_scaled", np.zeros((len(best.labels), 2)))[:, 1] if results.get("_X_scaled", np.zeros((1, 2))).shape[1] > 1 else np.zeros(len(best.labels)),
                c=best.labels,
                cmap="viridis",
                alpha=0.6,
                s=30,
            )
            ax.set_xlabel("Feature 1 (scaled)")
            ax.set_ylabel("Feature 2 (scaled)")
            ax.set_title(f"{best.name} – {best.n_clusters} clusters")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.info("Could not generate scatter plot.")

    # Download cluster labels
    if best.labels is not None and len(best.labels) > 0:
        labels_df = pd.DataFrame({
            "cluster_label": best.labels,
        })
        csv = labels_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cluster Labels as CSV",
            csv,
            file_name=f"{best.name.lower().replace(' ', '_')}_cluster_labels.csv",
            mime="text/csv",
            key="download_cluster_labels",
        )

def display_reduction_results(results: Dict) -> None:
    """Display dimensionality reduction results with scatter plot."""
    best = results.get("best")
    comparison = results.get("comparison")

    if best is None:
        st.error("No dimensionality reduction results available.")
        return

    st.subheader("Dimensionality Reduction Results")
    st.write(f"**Best Algorithm:** {best.name}")

    # Metrics
    if best.metrics:
        metrics_display = {k: v for k, v in best.metrics.items() if isinstance(v, (int, float))}
        if metrics_display:
            metric_cols = st.columns(min(len(metrics_display), 4))
            for i, (k, v) in enumerate(metrics_display.items()):
                with metric_cols[i % len(metric_cols)]:
                    display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    st.metric(k.replace("_", " ").title(), display_val)

    # Comparison table
    if comparison is not None and not comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

    # Scatter plot of reduced dimensions
    if best.transformed is not None and best.transformed.size > 0 and best.transformed.shape[1] >= 2:
        st.subheader("2D Projection")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                best.transformed[:, 0],
                best.transformed[:, 1],
                alpha=0.6,
                s=30,
                c="steelblue",
            )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"{best.name} – 2D Projection")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.info("Could not generate scatter plot.")

    # Download reduced data
    if best.transformed is not None and best.transformed.size > 0:
        reduced_df = pd.DataFrame(
            best.transformed,
            columns=[f"Component_{i+1}" for i in range(best.transformed.shape[1])],
        )
        csv = reduced_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Reduced Data as CSV",
            csv,
            file_name="reduced_data.csv",
            mime="text/csv",
        )


def display_anomaly_results(results: Dict) -> None:
    """Display anomaly detection results."""
    best = results.get("best")
    comparison = results.get("comparison")

    if best is None:
        st.error("No anomaly detection results available.")
        return

    st.subheader("Anomaly Detection Results")
    st.write(f"**Best Algorithm:** {best.name}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(best.labels) if best.labels is not None else 0)
    with col2:
        st.metric("Anomalies Found", best.n_anomalies)
    with col3:
        st.metric("Anomaly %", f"{best.anomaly_pct:.1f}%")

    # Comparison table
    if comparison is not None and not comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

    # Score distribution
    if best.scores is not None and len(best.scores) > 0:
        st.subheader("Anomaly Score Distribution")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            if best.labels is not None and len(best.labels) == len(best.scores):
                normal_scores = best.scores[best.labels == 1]
                anomaly_scores = best.scores[best.labels == -1]
                if len(normal_scores) > 0:
                    ax.hist(normal_scores, bins=30, alpha=0.6, label="Normal", color="steelblue")
                if len(anomaly_scores) > 0:
                    ax.hist(anomaly_scores, bins=30, alpha=0.6, label="Anomaly", color="crimson")
                ax.legend()
            else:
                ax.hist(best.scores, bins=30, alpha=0.7, color="steelblue")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Count")
            ax.set_title(f"{best.name} – Score Distribution")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.info("Could not generate score distribution plot.")

    # Download anomaly labels
    if best.labels is not None and len(best.labels) > 0:
        labels_df = pd.DataFrame({
            "anomaly_label": best.labels,
            "anomaly_score": best.scores if best.scores is not None and len(best.scores) == len(best.labels) else np.nan,
            "is_anomaly": best.labels == -1,
        })
        csv = labels_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Anomaly Labels as CSV",
            csv,
            file_name="anomaly_labels.csv",
            mime="text/csv",
        )


def run_unsupervised_flow(df: pd.DataFrame) -> None:
    """Run the full unsupervised learning workflow."""

    # Task selection
    unsupervised_task = st.selectbox(
        "Select unsupervised task",
        options=["Clustering", "Dimensionality Reduction", "Anomaly Detection"],
        index=0,
    )

    if unsupervised_task == "Clustering":
        st.info("**Use Case:** Group similar data points together without predefined labels. Best for customer segmentation, finding hidden patterns, or organizing unstructured data into distinct groups.")
    elif unsupervised_task == "Dimensionality Reduction":
        st.info("**Use Case:** Reduce the number of columns in your dataset while keeping its core structure. Best for visualizing high-dimensional data in 2D, removing noise, or compressing data.")
    elif unsupervised_task == "Anomaly Detection":
        st.info("**Use Case:** Identify rare items or events that differ significantly from the rest of the data. Best for fraud detection, finding outliers, or identifying faulty equipment.")

    task_key = unsupervised_task.lower().replace(" ", "_")

    # Column selection
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to use (leave empty to use all)",
        options=all_columns,
        default=[],
        help="Choose which columns the algorithm will analyze. Leave empty for all columns.",
    )
    columns_to_use = selected_columns if selected_columns else None

    # Mode selection
    unsup_mode = st.sidebar.radio(
        "Select mode",
        options=["Automatic (All Models)", "Manual (Choose Algorithm)"],
        index=0,
        key="unsup_mode",
    )

    selected_model_name = None
    manual_params = None

    if unsup_mode == "Manual (Choose Algorithm)":
        if task_key == "clustering":
            model_names = list(get_clustering_models().keys())
        elif task_key == "dimensionality_reduction":
            model_names = list(get_reduction_models().keys())
        else:
            model_names = list(get_anomaly_models().keys())

        selected_model_name = st.sidebar.selectbox(
            "Choose algorithm",
            options=model_names,
            key="unsup_model_select",
        )
        manual_params = get_unsupervised_manual_params(task_key, selected_model_name)

    # Run button
    button_label = f"Run {unsupervised_task}"
    if st.button(button_label):
        with st.spinner(f"Running {unsupervised_task.lower()}. This may take a moment..."):
            try:
                output = run_unsupervised_workflow(
                    df=df,
                    task=task_key,
                    columns=columns_to_use,
                    model_name=selected_model_name,
                    params=manual_params,
                )

                # Store preprocessed data for visualization
                if task_key == "clustering":
                    from src.superml_forge.unsupervised.preprocessor import preprocess_for_unsupervised
                    X_scaled, _, _ = preprocess_for_unsupervised(df, columns_to_use)
                    output["_X_scaled"] = X_scaled

                st.session_state["unsupervised_results"] = {
                    "task": task_key,
                    "output": output,
                }
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return

    # Display results
    if st.session_state.get("unsupervised_results"):
        cached = st.session_state["unsupervised_results"]
        if cached["task"] == "clustering":
            display_clustering_results(cached["output"])
        elif cached["task"] == "dimensionality_reduction":
            display_reduction_results(cached["output"])
        elif cached["task"] == "anomaly_detection":
            display_anomaly_results(cached["output"])


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════

def main() -> None:
    st.title("AutoML – End-to-End Automated Machine Learning")
    st.markdown(
        """
        Upload your tabular dataset and choose between **Supervised** or **Unsupervised** learning.
        The app will preprocess data, train models, and show you the best results.
        """
    )

    # ─── Sidebar ───
    st.sidebar.title("AutoML Settings")
    st.sidebar.markdown(
        textwrap.dedent(
            """
            Upload a dataset and choose your workflow:

            - **Supervised**: Classification & Regression with target column
            - **Unsupervised**: Clustering, Dimensionality Reduction, Anomaly Detection
            """
        )
    )

    # Learning type selector (top of sidebar)
    learning_type = st.sidebar.radio(
        "Learning Type",
        options=["Supervised Learning", "Unsupervised Learning"],
        index=0,
        key="learning_type",
    )

    # ─── File upload ───
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "tsv", "json", "parquet", "txt"],
    )
    if uploaded_file is None:
        st.info("Please upload a tabular file to get started.")
        return

    file_type = detect_extension(uploaded_file.name)
    st.caption(f"Detected file type: `{file_type}`")

    selected_sheet = None
    if file_type in {".xlsx", ".xls"}:
        try:
            sheets = list_excel_sheets(uploaded_file)
            selected_sheet = st.selectbox("Select Excel sheet", options=sheets, index=0)
        except Exception as exc:
            st.error(f"Could not read Excel sheets: {exc}")
            return

    # Load and preview dataset
    try:
        df = cached_load_data(uploaded_file, selected_sheet)
        df = sanitize_dataframe(df)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    if df.empty:
        st.error("The uploaded dataset appears to be empty.")
        return

    st.session_state["loaded_df"] = df.copy()
    dataset_info = get_dataset_info(df)
    display_dataset_info(dataset_info)
    show_missing_value_summary(df)

    csv_export = df.to_csv(index=False).encode("utf-8")
    export_name = f"{Path(uploaded_file.name).stem}_cleaned.csv"
    st.download_button("Download cleaned data as CSV", csv_export, file_name=export_name, mime="text/csv")

    # ─── Branch based on learning type ───
    st.divider()

    if learning_type == "Supervised Learning":
        st.header("🎯 Supervised Learning")
        run_supervised_flow(df, uploaded_file)
    else:
        st.header("🔍 Unsupervised Learning")
        run_unsupervised_flow(df)


if __name__ == "__main__":
    main()
