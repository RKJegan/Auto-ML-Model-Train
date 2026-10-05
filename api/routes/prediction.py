"""Prediction endpoints routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//prediction")
async def prediction_endpoint():
    """GET /prediction"""
    return {"status": "ok", "endpoint": "prediction"}
