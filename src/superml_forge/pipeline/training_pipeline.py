"""End-to-end training pipeline."""
from __future__ import annotations


class TrainingPipeline:
    """Orchestrates the full training workflow: ingest → validate → profile → preprocess → split → train → evaluate → select."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def run(self, data_path: str, target_column: str) -> dict:
        """Run the full training pipeline. Returns a summary dict."""
        # TODO: wire up all workflow stages
        return {"status": "not_implemented"}
