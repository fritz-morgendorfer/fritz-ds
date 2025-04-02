from typing import Literal, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]
DatasetType = Literal["train", "validation", "test"]
