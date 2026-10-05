"""Report generation routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("//reports")
async def reports_endpoint():
    """GET /reports"""
    return {"status": "ok", "endpoint": "reports"}
