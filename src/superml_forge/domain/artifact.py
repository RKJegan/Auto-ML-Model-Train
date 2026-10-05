"""Artifact domain object."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArtifactInfo:
    """Metadata about a saved artifact."""
    name: str
    path: str
    artifact_type: str  # "model", "preprocessor", "encoder", etc.
    version: str = "1.0"
