# AutoML (SuperML-Forge) 🚀

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20FastAPI-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Container](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, full-stack **Automated Machine Learning (AutoML)** platform and interactive web application designed for tabular data. **SuperML-Forge** orchestrates the complete machine learning lifecycle from ingestion and intelligent data cleaning, to automated feature selection, hyperparameter tuning, model evaluation, artifact serialization, interactive web-based inference, and RESTful API serving.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
  - [Supervised Learning Engine](#1-supervised-learning-engine)
  - [Unsupervised Learning Suite](#2-unsupervised-learning-suite)
  - [Robust Preprocessing & Quality Assurance](#3-robust-preprocessing--quality-assurance)
  - [Intelligent Feature Engineering & Selection](#4-intelligent-feature-engineering--selection)
  - [Interactive Prediction & Model Export](#5-interactive-prediction--model-export)
- [Supported Machine Learning Algorithms](#-supported-machine-learning-algorithms)
- [System Architecture & Directory Structure](#-system-architecture--directory-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [Running the Application](#-running-the-application)
  - [1. Streamlit Web UI](#1-streamlit-interactive-web-dashboard)
  - [2. FastAPI RESTful Service](#2-fastapi-rest-service)
  - [3. Command-Line Interface (CLI)](#3-command-line-interface-cli)
- [Docker & Containerized Deployment](#-docker--containerized-deployment)
- [Deploying to Cloud (Render via Docker)](#-deploying-to-cloud-render-via-docker)
- [API Documentation & Endpoints](#-api-documentation--endpoints)
- [Configuration Management](#-configuration-management)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Contributing & License](#-contributing--license)

---

## 🏛 Overview & Architecture

SuperML-Forge is engineered with a modular, decoupled architecture that bridges production-grade machine learning pipelines with an intuitive, responsive user interface.

```
                              ┌────────────────────────────────────────┐
                              │             Tabular Data               │
                              │ (CSV, Excel, TSV, JSON, Parquet, TXT) │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │       Ingestion & Validation Engine    │
                              │   - Multi-format parsing               │
                              │   - pd.NA / NaN Sanitization           │
                              │   - Whitespace & Dtype Normalization   │
                              └───────────────────┬────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │       Supervised Workflow        │             │      Unsupervised Workflow       │
        ├──────────────────────────────────┤             ├──────────────────────────────────┤
        │ • Target & Problem Detection     │             │ • Task: Cluster / Dim-Red / Anom │
        │ • Feature Selection (ID Drops)   │             │ • Label Encoding + Numeric Conv  │
        │ • Central Tendency Imputation    │             │ • Median Imputation & Scaling    │
        │ • One-Hot Encoding (Ignore Unk)  │             │ • Model Benchmark & Evaluation   │
        │ • Top 75% ANOVA Feature Parsing  │             │ • 2D Scatter Projections         │
        │ • RandomizedSearchCV / Tuning    │             │ • Cluster / Anomaly CSV Export   │
        │ • Model Comparison Leaderboard   │             └──────────────────────────────────┘
        │ • Model Export (.pkl) & Predict  │
        └──────────────────────────────────┘
```

---

## 🌟 Key Features

### 1. Supervised Learning Engine
- **Automatic Task Detection**: Inspects the target column distribution, cardinality, and data types to dynamically categorize tasks into **Classification** or **Regression**.
- **Model Comparison Leaderboard**: Concurrently fits baseline and candidate models, benchmarked via cross-validation, displaying comparative performance metrics side-by-side.
- **Automated Hyperparameter Optimization**: Leverages Scikit-Learn's `RandomizedSearchCV` across predefined search spaces for optimal parameter extraction.
- **Manual Parameter Tuning Mode**: Allows data scientists and engineers to manually override models and configure hyperparameters (e.g., regularization strength $C$, tree depth, criteria, solvers) directly via UI sliders and dropdowns.
- **Comprehensive Evaluation Metrics**:
  - *Classification*: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and Seaborn Confusion Matrix heatmaps.
  - *Regression*: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).
  - *Explainability*: Dynamic extraction of tree-based feature importances or linear model coefficients.

### 2. Unsupervised Learning Suite
- **Clustering**: Identifies natural groupings within your data with automatic cluster scoring using **Silhouette Score**, **Calinski-Harabasz Index**, and **Davies-Bouldin Index**. Generates 2D cluster scatter visualizations.
- **Dimensionality Reduction**: Projects high-dimensional datasets into 2D manifolds for visual exploration and variance interpretation.
- **Anomaly & Outlier Detection**: Discovers latent anomalies using multivariate density and boundary estimators, calculating anomaly contamination percentages and score distributions.
- **Label & Projection Export**: One-click download of generated cluster assignments, dimensionally reduced components, or outlier flags as structured CSVs.

### 3. Robust Preprocessing & Quality Assurance
- **Central Tendency Imputation**: Prevents data loss caused by row deletion.
  - *Numerical Features*: Handled via **Median** imputation, resilient against heavy tails and extreme outliers.
  - *Categorical Features*: Handled via **Statistical Mode** (`most_frequent`), preserving distribution fidelity.
- **Zero-Crash Pandas Extension Handling (`pd.NA`)**: Specialized sanitizer that strips ambigious `pd.NA` missing indicators and converts non-standard extension arrays into standard NumPy-compatible representations before pipeline ingestion.
- **Robust One-Hot Encoding**: Fitted with `handle_unknown="ignore"` to ensure zero runtime inference failures when unseen categorical levels appear during real-time prediction.

### 4. Intelligent Feature Engineering & Selection
- **Automated ID & Identifier Elimination**: Automatically drops high-risk, zero-signal identifiers (e.g., `id`, `customerid`, `passengerid`, `guid`, `uuid`).
- **100% Cardinality Detection**: Identifies and drops numerical columns where every record has a unique value (index columns disguised as numerical features).
- **High-Cardinality Categorical Purging**: Drops text columns where unique entries exceed 50% of total rows (e.g., customer names, ticket IDs), preventing categorical explosion.
- **ANOVA F-Value Feature Parsing**: Pipelines integrate `SelectPercentile(percentile=75)` using ANOVA F-tests (`f_classif` or `f_regression`) to drop noisy, irrelevant features and accelerate inference.

### 5. Interactive Prediction & Model Export
- **Dynamic Prediction Form**: Generates an adaptive input form in the web UI. It strictly displays the selected, engineered features—excluding dropped ID columns.
- **Serialized Model Download**: Direct download of the trained full pipeline as a portable `joblib` (`.pkl`) artifact, ready for drop-in deployment with `joblib.load()`.

---

## 🤖 Supported Machine Learning Algorithms

### Supervised Learning
| Domain | Algorithm | Scikit-Learn Class | Primary Use Case |
|---|---|---|---|
| **Classification** | **Logistic Regression** | `LogisticRegression` | Fast, interpretable linear classification |
| **Classification** | **Decision Tree** | `DecisionTreeClassifier` | Non-linear decision rules and segment analysis |
| **Classification** | **Random Forest** | `RandomForestClassifier` | Ensemble bagging for high accuracy and variance reduction |
| **Classification** | **Support Vector Machine (SVM)**| `SVC(probability=True)` | High-dimensional margin classification |
| **Regression** | **Linear Regression** | `LinearRegression` | Baseline continuous target estimation |
| **Regression** | **Ridge Regression** | `Ridge` | $L_2$-regularized linear regression for multicollinearity |
| **Regression** | **Lasso Regression** | `Lasso` | $L_1$-regularized regression with sparse feature weights |
| **Regression** | **Random Forest Regressor** | `RandomForestRegressor` | Ensemble non-linear regression |

### Unsupervised Learning
| Domain | Algorithm | Primary Metrics | Output Artifact |
|---|---|---|---|
| **Clustering** | **K-Means** | Silhouette, Calinski-Harabasz, Davies-Bouldin | Discrete Cluster IDs |
| **Clustering** | **Mini-Batch K-Means** | Silhouette Score | Large-scale Cluster IDs |
| **Clustering** | **DBSCAN** | Density-based cluster count | Noise & Cluster IDs |
| **Clustering** | **Agglomerative Clustering** | Dendrogram-based distance | Hierarchical Cluster IDs |
| **Clustering** | **Mean Shift** | Bandwidth estimation | Centroid Cluster IDs |
| **Clustering** | **Gaussian Mixture Models (GMM)** | Log-likelihood, BIC/AIC | Probabilistic Clusters |
| **Dim. Reduction** | **PCA** | Explained Variance Ratio | Orthogonal Components |
| **Dim. Reduction** | **t-SNE** | 2D Manifold Embedding | Non-linear 2D Coordinates |
| **Dim. Reduction** | **Truncated SVD** | Variance Explained | Sparse Matrix Components |
| **Anomaly Detection**| **Isolation Forest** | Anomaly Score, Contamination % | Binary Anomaly Labels |
| **Anomaly Detection**| **Local Outlier Factor (LOF)** | Local density factor | Outlier Identification |
| **Anomaly Detection**| **One-Class SVM** | Support boundary distance | Novelty/Outlier Flags |

---

## 📂 System Architecture & Directory Structure

```
Auto-ML-Model-Train/
├── app.py                             # Main Streamlit web dashboard
├── Dockerfile                         # Production Docker container definition
├── docker-compose.yml                 # Multi-container service specification
├── Makefile                           # Developer CLI shortcuts & build automation
├── pyproject.toml                     # Python packaging and linter configs
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Testing, linting, and development dependencies
│
├── api/                               # FastAPI REST API Service
│   ├── main.py                        # API application factory & routing
│   ├── dependencies.py                # Shared dependency injection
│   ├── middleware.py                  # CORS and logging middleware
│   ├── routes/                        # REST endpoint controllers
│   │   ├── health.py                  # Service heartbeat check
│   │   ├── datasets.py                # Dataset upload and validation
│   │   ├── training.py                # Asynchronous training triggers
│   │   ├── models.py                  # Model registry & metadata inspection
│   │   ├── prediction.py              # Real-time JSON inference
│   │   └── reports.py                 # Evaluation report generation
│   ├── schemas/                       # Pydantic request/response models
│   └── services/                      # API business logic adapters
│
├── configs/                           # Modular YAML Configurations
│   ├── config.yaml                    # Global platform configuration
│   ├── models.yaml                    # Model hyperparameter boundaries
│   ├── training.yaml                  # Cross-validation folds and iterations
│   ├── preprocessing.yaml             # Imputation strategies and scaling types
│   ├── evaluation.yaml                # Performance thresholds and metrics
│   └── logging.yaml                   # Logging formatters and handlers
│
├── src/superml_forge/                 # Core Machine Learning Framework
│   ├── ingestion/                     # Multi-format tabular data loaders
│   ├── validation/                    # Data quality sanitizers & pd.NA repair
│   ├── profiling/                     # Exploratory statistics & column profiling
│   ├── task_detection/                # Automatic Classification/Regression detector
│   ├── feature_engineering/           # Automated identifier & noise filtering
│   ├── preprocessing/                 # Pipelines, transformers, and imputers
│   │   └── transformers/              # Custom scikit-learn transformers
│   ├── splitting/                     # Stratified & random train-test splitting
│   ├── models/                        # Model definitions & registry maps
│   ├── training/                      # Training loop & RandomizedSearchCV execution
│   ├── tuning/                        # Search spaces & distribution generators
│   ├── evaluation/                    # Metrics, confusion matrices, feature importance
│   ├── selection/                     # Best-model selection criteria
│   ├── prediction/                    # Inference pipelines
│   ├── reporting/                     # Automated report generators
│   └── unsupervised/                  # Unsupervised Learning Subsystem
│       ├── clustering/                # K-Means, DBSCAN, Agglomerative, GMM
│       ├── dimensionality_reduction/  # PCA, t-SNE, TruncatedSVD
│       ├── anomaly_detection/         # Isolation Forest, LOF, One-Class SVM
│       ├── preprocessor.py            # Unsupervised scaling & label encoding
│       └── workflow.py                # Unsupervised orchestration workflow
│
├── scripts/                           # Standalone CLI utilities
│   ├── train.py                       # CLI model trainer
│   ├── predict.py                     # CLI batch inference runner
│   ├── evaluate.py                    # CLI model evaluator
│   └── profile.py                     # CLI dataset profiler
│
├── data/                              # Local dataset storage (gitignored)
├── artifacts/                         # Serialized models and exports (gitignored)
└── tests/                             # Comprehensive Automated Test Suite
    ├── unit/                          # Unit tests for preprocessing & models
    ├── integration/                   # Pipeline integration tests
    ├── api/                           # FastAPI test client test suite
    └── test_all_flows.py              # End-to-end integration verification
```

---

## 🛠️ Installation & Local Setup

### Prerequisites
- Python 3.10 or 3.11 installed
- Git
- (Optional) Docker Desktop

### 1. Clone Repository
```bash
git clone https://github.com/RKJegan/Auto-ML-Model-Train.git
cd Auto-ML-Model-Train
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: Install developer tooling for testing and linting)*
```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Running the Application

### 1. Streamlit Interactive Web Dashboard
Launch the primary AutoML user interface:
```bash
streamlit run app.py
```
*Open your browser at `http://localhost:8501`.*

### 2. FastAPI REST Service
Launch the backend REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Access interactive Swagger API documentation at `http://localhost:8000/docs`.*

### 3. Command-Line Interface (CLI)
Train models directly from your terminal:
```bash
python scripts/train.py --data "data/dataset.csv" --target "target_column" --output "artifacts/models"
```

---

## 🐳 Docker & Containerized Deployment

The repository includes an optimized `Dockerfile` based on `python:3.11-slim` with layer caching.

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t auto-ml-forge .

# Run the container (exposing Streamlit on 8501 and FastAPI on 8000)
docker run -p 8501:8501 -p 8000:8000 auto-ml-forge
```

### Run with Docker Compose
Mount volumes for datasets, artifacts, and logs automatically:
```bash
docker-compose up --build -d
```
To stop the services:
```bash
docker-compose down
```

---

## 🌐 Deploying to Cloud (Render via Docker)

This application can be hosted online for free using [Render](https://render.com/).

### Step-by-Step Guide
1. **Fork or Push**: Ensure your project is on GitHub at `https://github.com/RKJegan/Auto-ML-Model-Train`.
2. **Sign In to Render**: Navigate to [Render.com](https://render.com/) and link your GitHub account.
3. **Create Web Service**:
   - In your dashboard, click **New +** ➡️ **Web Service**.
   - Select your repository: `RKJegan/Auto-ML-Model-Train`.
4. **Configure Service Details**:
   - **Name**: `auto-ml-forge` (or any custom identifier)
   - **Region**: Select the region closest to you
   - **Branch**: `main`
   - **Root Directory**: *(Leave empty)*
   - **Environment / Runtime**: Select **`Docker`** ⚠️ *(Crucial: do not select Python)*
   - **Instance Type**: Select the **Free** tier
5. **Deploy**:
   - Click **Create Web Service**.
   - Render reads your `Dockerfile`, installs dependencies, and boots Streamlit on port `8501`.
   - Within 2-3 minutes, your live public URL (e.g., `https://auto-ml-forge.onrender.com`) will be available.

---

## 📡 API Documentation & Endpoints

When running the FastAPI server (`uvicorn api.main:app`), the following endpoints are available:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Heartbeat & service health status |
| `POST` | `/api/v1/datasets/upload` | Upload and validate raw tabular files |
| `POST` | `/api/v1/training/train` | Trigger an asynchronous training pipeline |
| `GET` | `/api/v1/models/` | List all trained models in registry |
| `POST` | `/api/v1/prediction/predict` | Real-time prediction given feature JSON payload |
| `GET` | `/api/v1/reports/{model_id}` | Retrieve JSON/Markdown performance evaluation reports |

Explore the live OpenAPI schema at `http://localhost:8000/docs` or `http://localhost:8000/redoc`.

---

## ⚙️ Configuration Management

All system defaults and training parameters can be customized without editing code via `configs/`:

- `configs/config.yaml`: Global defaults, random seed states, and temporary paths.
- `configs/preprocessing.yaml`: Missing value imputation strategies, categorical thresholds, scaling standard.
- `configs/training.yaml`: Cross-validation folds (`cv=3` or `cv=5`), iteration counts (`n_iter=10`), parallel workers (`n_jobs=-1`).
- `configs/models.yaml`: Parameter search grids for all classification and regression estimators.
- `configs/logging.yaml`: Logging formatting, log rotation, and stream output levels.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite using `pytest`:

```bash
# Run all unit and integration tests
pytest tests/ -v --tb=short

# Run only the end-to-end flow test
pytest tests/test_all_flows.py -v

# Run linting checks
flake8 src/ api/ scripts/
mypy src/ api/
```

Or execute shortcuts via the `Makefile`:
```bash
make test     # Runs pytest suite
make lint     # Runs flake8 and mypy
make format   # Automatically formats code with black and isort
make clean    # Cleans cached bytecode and test artifacts
```

---

## 📄 License & Credits

- **Author**: Jegan R K
- **Repository**: [https://github.com/RKJegan/Auto-ML-Model-Train](https://github.com/RKJegan/Auto-ML-Model-Train)
- **License**: Distributed under the **MIT License**. See [LICENSE](LICENSE) for full legal text.
