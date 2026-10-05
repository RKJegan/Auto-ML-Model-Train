# AutoML (SuperML-Forge)

An end-to-end **Automated Machine Learning** web application for both **supervised** and **unsupervised** learning tasks.
Upload a tabular dataset, and the app will automatically preprocess data, train multiple models,
and show you the best results with metrics and visualizations.

## Features

### Supervised Learning
- **Multi-format data ingestion**: CSV, Excel, TSV, JSON, Parquet, TXT
- **Automatic problem detection**: Classification vs Regression
- **Smart preprocessing**: Missing value imputation, categorical encoding, numerical scaling
- **Model training & tuning**: Multiple algorithms with RandomizedSearchCV
- **Evaluation metrics**: Accuracy, Precision, Recall, F1 (classification); RMSE, MAE, R² (regression)
- **Interactive prediction**: Predict new values through the web UI
- **Dual mode**: Automatic (tries all models) or Manual (pick your algorithm)

### Unsupervised Learning
- **Clustering**: K-Means, Mini-Batch K-Means, DBSCAN, Agglomerative, Mean Shift, Gaussian Mixture
- **Dimensionality Reduction**: PCA, t-SNE, Truncated SVD
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM
- **Metrics**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin, Explained Variance, Anomaly %
- **Visualizations**: Cluster scatter plots, 2D projections, score distributions
- **Download results**: Export cluster labels, reduced data, anomaly flags as CSV

## Supported Models

### Supervised

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree | Ridge Regression |
| Random Forest | Lasso Regression |
| SVM | Random Forest Regressor |

### Unsupervised

| Clustering | Dim. Reduction | Anomaly Detection |
|---|---|---|
| K-Means | PCA | Isolation Forest |
| Mini-Batch K-Means | t-SNE | Local Outlier Factor |
| DBSCAN | Truncated SVD | One-Class SVM |
| Agglomerative Clustering | | |
| Mean Shift | | |
| Gaussian Mixture | | |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
SuperML-Forge/
├── app.py                    # Streamlit web application
├── src/superml_forge/        # Core ML package
│   ├── ingestion/            # Data loading
│   ├── validation/           # Data quality checks
│   ├── profiling/            # Dataset statistics
│   ├── task_detection/       # Classification/Regression detection
│   ├── preprocessing/        # Imputation, encoding, scaling
│   ├── splitting/            # Train/test split
│   ├── models/               # Supervised model definitions & registry
│   ├── training/             # Training loop & tuning
│   ├── tuning/               # Hyperparameter search spaces
│   ├── evaluation/           # Supervised metrics & feature importance
│   ├── selection/            # Best model selection
│   ├── prediction/           # Inference
│   ├── unsupervised/         # Unsupervised learning
│   │   ├── clustering/       # K-Means, DBSCAN, etc.
│   │   ├── dimensionality_reduction/  # PCA, t-SNE, etc.
│   │   ├── anomaly_detection/  # Isolation Forest, LOF, etc.
│   │   └── evaluation/       # Clustering & anomaly metrics
│   ├── pipeline/             # End-to-end orchestration
│   └── utils/                # Logging, exceptions, helpers
├── api/                      # FastAPI REST API
├── configs/                  # YAML configuration files
├── data/                     # Dataset storage
├── artifacts/                # Trained model storage
├── tests/                    # Unit & integration tests
├── scripts/                  # CLI entry points
├── notebooks/                # Exploration notebooks
├── docs/                     # Documentation
└── deployment/               # Docker & monitoring
```

## How It Works

### Supervised Learning
1. **Upload** a tabular dataset
2. **Select** the target column
3. **Detect** problem type automatically (Classification / Regression)
4. **Train** multiple models with cross-validation and hyperparameter tuning
5. **Evaluate** and select the best model
6. **Predict** new values interactively

### Unsupervised Learning
1. **Upload** a tabular dataset
2. **Choose** task: Clustering / Dimensionality Reduction / Anomaly Detection
3. **Select** columns (optional) and algorithm mode (Automatic / Manual)
4. **Run** the analysis
5. **View** results: cluster labels, 2D plots, anomaly scores
6. **Download** results as CSV

## License

MIT License – see [LICENSE](LICENSE) for details.
