"""Model management routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//models")
async def models_endpoint():
    """GET /models"""
    return {"status": "ok", "endpoint": "models"}
