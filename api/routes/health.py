"""Health check routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//health")
async def health_endpoint():
    """GET /health"""
    return {"status": "ok", "endpoint": "health"}
