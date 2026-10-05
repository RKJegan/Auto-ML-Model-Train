"""Feature domain object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureSet:
    """Describes the features used for training."""
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)

    @property
    def all_features(self) -> List[str]:
        return self.numeric + self.categorical
