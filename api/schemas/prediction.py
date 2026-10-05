"""Prediction schemas."""
from __future__ import annotations

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request schema for prediction."""
    pass


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    status: str = "ok"
