"""Validation set splitting (scaffold)."""
from __future__ import annotations


def create_validation_split(X_train, y_train, val_size: float = 0.1, random_state: int = 42):
    """Split training data into train and validation sets."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
