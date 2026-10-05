"""Pipeline configuration (scaffold)."""
from __future__ import annotations


def load_pipeline_config(path: str = "configs/config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
