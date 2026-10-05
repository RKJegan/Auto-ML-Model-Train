"""Response schemas."""
from __future__ import annotations

from pydantic import BaseModel


class ResponseRequest(BaseModel):
    """Request schema for response."""
    pass


class ResponseResponse(BaseModel):
    """Response schema for response."""
    status: str = "ok"
