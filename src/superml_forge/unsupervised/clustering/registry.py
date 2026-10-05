"""Registry of clustering algorithms."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture


def get_clustering_models(
    n_clusters: int = 3, random_state: int = 42
) -> Dict[str, Any]:
    """Return a dictionary of clustering model instances."""
    return {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        "Mini-Batch K-Means": MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "Mean Shift": MeanShift(),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=random_state),
    }


def get_clustering_param_options() -> Dict[str, Dict[str, Any]]:
    """Return hyperparameter options for each clustering algorithm."""
    return {
        "K-Means": {
            "n_clusters": {"type": "int", "min": 2, "max": 20, "default": 3},
            "init": {"type": "select", "options": ["k-means++", "random"], "default": "k-means++"},
            "max_iter": {"type": "int", "min": 100, "max": 1000, "default": 300},
        },
        "Mini-Batch K-Means": {
            "n_clusters": {"type": "int", "min": 2, "max": 20, "default": 3},
            "batch_size": {"type": "int", "min": 50, "max": 5000, "default": 1024},
        },
        "DBSCAN": {
            "eps": {"type": "float", "min": 0.01, "max": 10.0, "default": 0.5},
            "min_samples": {"type": "int", "min": 2, "max": 50, "default": 5},
        },
        "Agglomerative Clustering": {
            "n_clusters": {"type": "int", "min": 2, "max": 20, "default": 3},
            "linkage": {"type": "select", "options": ["ward", "complete", "average", "single"], "default": "ward"},
        },
        "Mean Shift": {
            "bandwidth": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "note": "Leave 0 for auto-estimation"},
        },
        "Gaussian Mixture": {
            "n_components": {"type": "int", "min": 2, "max": 20, "default": 3},
            "covariance_type": {"type": "select", "options": ["full", "tied", "diag", "spherical"], "default": "full"},
        },
    }
