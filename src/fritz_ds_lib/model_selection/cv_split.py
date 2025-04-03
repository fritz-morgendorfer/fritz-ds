from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.core.names import ArrayLike


@runtime_checkable
class SklearnCvSplitProtocol(Protocol):
    @abstractmethod
    def split(self, X: ArrayLike, y: ArrayLike = None, groups: ArrayLike = None):
        """Generate indices to split data into training and test set."""

    @abstractmethod
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""


class RollingTimeSeriesSplit(ProjectBaseModel):
    max_period: datetime
    periods: int
    freq: str
    col_date: str

    @property
    def _fcst_timestamps(self) -> list[datetime]:
        ts = list(
            pd.date_range(end=self.max_period, periods=self.periods, freq=self.freq)
        )
        return ts

    def split(self, X: ArrayLike, y: ArrayLike = None, groups: ArrayLike = None):
        """Generate indices to split data into training and test set."""
        groups = X[self.col_date]
        for ts in self._fcst_timestamps:
            train_idx = groups < ts
            test_idx = groups == ts
            yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return len(self._fcst_timestamps)
