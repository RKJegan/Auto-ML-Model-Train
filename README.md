# AutoML (SuperML-Forge) 🚀

An end-to-end, fully robust **Automated Machine Learning** web application built with Streamlit and Scikit-Learn. The platform handles both **supervised** and **unsupervised** learning tasks with enterprise-grade data preprocessing, automated feature engineering, and one-click deployment capabilities.

Upload any tabular dataset, and the app will automatically clean the data, engineer features, train multiple models, and show you the best results with interactive predictions.

---

## 🌟 Key Features

### 1. Robust Data Preprocessing Engine
- **Missing Value Central Tendency Imputation**: Instead of blindly dropping rows with missing values, the system intelligently imputes them based on statistical central tendency:
  - *Numeric Features*: Uses the **Median** to prevent outliers from skewing the data.
  - *Categorical Features*: Uses the **Statistical Mode** (most frequent value) to accurately represent missing classes.
- **Advanced Null Handling (`pd.NA`)**: Safely parses and normalizes pandas extension types and string anomalies that typically crash standard machine learning pipelines.
- **Categorical Encoding**: Automatically applies One-Hot Encoding with safe `handle_unknown="ignore"` logic to prevent inference crashes on unseen data.

### 2. Automated Feature Engineering & Selection
- **Intelligent ID Dropping**: The system scans your dataset and automatically drops irrelevant identifiers (e.g., `customerid`, `uuid`, `passengerid`) *before* they can confuse the model.
- **High-Cardinality Filtering**: Automatically filters out text columns (like Names or Tickets) where >50% of the values are unique, reducing noise.
- **100% Unique Numeric Dropping**: Identifies and drops numerical indexes pretending to be features.
- **ANOVA F-Value Selection**: After basic cleaning, the Scikit-Learn pipeline runs statistical tests to drop the bottom 25% of the most useless features, keeping only the top 75% most predictive features (`SelectPercentile`).

### 3. Smart Model Training & Tuning
- **Automatic Task Detection**: Dynamically detects whether your target column requires **Classification** or **Regression**.
- **Cross-Validated Search**: Tests multiple algorithms concurrently using `RandomizedSearchCV`.
- **Manual Override Mode**: Want to tune hyperparameters yourself? Use the UI to manually select algorithms and set learning rates, max depths, and penalties.

### 4. Interactive Prediction UI
Once the model is trained, a dynamic form is generated on the web page. Because of the upfront feature filtering, **the UI will only ask you for the relevant, important features**, hiding the useless IDs and names.

---

## 🤖 Supported Models

### Supervised Learning
| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree Classifier | Ridge Regression |
| Random Forest Classifier | Lasso Regression |
| XGBoost (Coming Soon) | Random Forest Regressor |

### Unsupervised Learning
| Clustering | Dimensionality Reduction | Anomaly Detection |
|---|---|---|
| K-Means | PCA | Isolation Forest |
| Mini-Batch K-Means | t-SNE | Local Outlier Factor |
| DBSCAN | Truncated SVD | One-Class SVM |
| Agglomerative Clustering | | |

---

## 🛠️ Installation & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/RKJegan/Auto-ML-Model-Train.git
cd Auto-ML-Model-Train

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit Application
streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 🐳 Deployment (Docker & Render)

This project includes a production-ready `Dockerfile` and can be hosted online for free using [Render](https://render.com/).

**How to deploy to Render:**
1. Go to **Render.com** and sign in with GitHub.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`RKJegan/Auto-ML-Model-Train`).
4. In the settings, set:
   - **Environment**: `Docker` *(Crucial: do not select Python)*
   - **Branch**: `main`
   - **Instance Type**: Free
5. Click **Create Web Service**. 

Render will automatically build the container from the `Dockerfile`, install the dependencies, and expose port `8501`. Within 3 minutes, your AutoML platform will be live on a public URL!

---

## 📂 Project Architecture

```
Auto-ML-Model-Train/
├── app.py                    # Main Streamlit UI & Orchestrator
├── Dockerfile                # Production Container definition
├── src/superml_forge/        # Core ML Engine
│   ├── feature_engineering/  # ID dropping & cardinality filtering
│   ├── validation/           # pd.NA handling & data quality
│   ├── preprocessing/        # Central tendency imputation & pipelines
│   ├── splitting/            # Train/test split logic
│   ├── training/             # RandomizedSearchCV loops
│   └── unsupervised/         # Clustering & Anomaly Detection
├── data/                     # Local dataset storage (gitignored)
└── tests/                    # System Tests
```

---

## 📜 License

MIT License – Copyright (c) Jegan R K. See [LICENSE](LICENSE) for details.
