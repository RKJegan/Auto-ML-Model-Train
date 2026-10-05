"""Feature Selector - Parses important features before training."""
from __future__ import annotations

import pandas as pd


def filter_important_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Parse only the important features.
    Drops obvious IDs and high-cardinality categorical features that are irrelevant.
    """
    X_filtered = X.copy()
    cols_to_drop = []
    
    id_keywords = ['id', 'customerid', 'passengerid', 'uuid', 'guid', 'index']
    
    for col in X_filtered.columns:
        col_lower = str(col).lower().strip()
        
        # 1. Drop explicit ID columns by name
        if col_lower in id_keywords:
            cols_to_drop.append(col)
            continue
            
        # 2. Drop numeric columns where every value is completely unique (100% cardinality)
        if pd.api.types.is_numeric_dtype(X_filtered[col]) and X_filtered[col].nunique() == len(X_filtered):
            cols_to_drop.append(col)
            continue
            
        # 3. Drop text columns (like names or tickets) where over 50% of values are unique
        if pd.api.types.is_object_dtype(X_filtered[col]) or pd.api.types.is_string_dtype(X_filtered[col]):
            if X_filtered[col].nunique() > 0.5 * len(X_filtered):
                cols_to_drop.append(col)
                continue
                
    X_filtered = X_filtered.drop(columns=cols_to_drop, errors='ignore')
    return X_filtered
