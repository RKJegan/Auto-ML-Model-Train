"""Registry of dimensionality reduction algorithms."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


def get_reduction_models(
    n_components: int = 2, random_state: int = 42
) -> Dict[str, Any]:
    """Return a dictionary of dimensionality reduction model instances."""
    return {
        "PCA": PCA(n_components=n_components, random_state=random_state),
        "t-SNE": TSNE(n_components=n_components, random_state=random_state, perplexity=30.0),
        "Truncated SVD": TruncatedSVD(n_components=n_components, random_state=random_state),
    }


def get_reduction_param_options() -> Dict[str, Dict[str, Any]]:
    """Return hyperparameter options for each reduction algorithm."""
    return {
        "PCA": {
            "n_components": {"type": "int", "min": 2, "max": 50, "default": 2},
        },
        "t-SNE": {
            "n_components": {"type": "int", "min": 2, "max": 3, "default": 2},
            "perplexity": {"type": "float", "min": 5.0, "max": 50.0, "default": 30.0},
            "learning_rate": {"type": "float", "min": 10.0, "max": 1000.0, "default": 200.0},
        },
        "Truncated SVD": {
            "n_components": {"type": "int", "min": 2, "max": 50, "default": 2},
        },
    }
