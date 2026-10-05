"""Training endpoints routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//training")
async def training_endpoint():
    """GET /training"""
    return {"status": "ok", "endpoint": "training"}
