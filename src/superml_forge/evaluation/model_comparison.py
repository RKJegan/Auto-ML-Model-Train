"""Model comparison utilities (scaffold)."""
from __future__ import annotations

import pandas as pd


def build_comparison_table(results: list) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of model results."""
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "Score": r.best_score,
            "Params": r.best_params,
        })
    return pd.DataFrame(rows)
