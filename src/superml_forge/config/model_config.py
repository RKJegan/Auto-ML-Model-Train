"""Model configuration (scaffold)."""
from __future__ import annotations


def load_model_config(path: str = "configs/models.yaml") -> dict:
    """Load model configuration from YAML."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
