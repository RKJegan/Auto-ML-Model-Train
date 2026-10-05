"""Training schemas."""
from __future__ import annotations

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    """Request schema for training."""
    pass


class TrainingResponse(BaseModel):
    """Response schema for training."""
    status: str = "ok"
