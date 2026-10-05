"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI

from .routes import health, training, prediction, models, datasets, reports

app = FastAPI(
    title="SupervisedAutoML API",
    version="0.1.0",
    description="REST API for SupervisedAutoML – automated machine learning for supervised tasks.",
)

app.include_router(health.router, tags=["Health"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])
