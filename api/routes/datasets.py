"""Dataset management routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//datasets")
async def datasets_endpoint():
    """GET /datasets"""
    return {"status": "ok", "endpoint": "datasets"}
