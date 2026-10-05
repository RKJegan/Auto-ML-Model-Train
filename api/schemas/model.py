"""Model schemas."""
from __future__ import annotations

from pydantic import BaseModel


class ModelRequest(BaseModel):
    """Request schema for model."""
    pass


class ModelResponse(BaseModel):
    """Response schema for model."""
    status: str = "ok"
