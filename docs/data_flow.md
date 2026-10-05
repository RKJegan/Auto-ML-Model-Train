# Data Flow

```
Upload → Ingestion → Validation → Profiling → Task Detection
  → Preprocessing → Splitting → Training → Tuning
  → Evaluation → Selection → Prediction
```

Data flows through the pipeline as pandas DataFrames. Each stage produces a result object that feeds into the next.
