from typing import Hashable, Literal, Sequence

import pandas as pd

REGIONS_MODULE = "regions.yaml"


def drop(
    df: pd.DataFrame,
    cols: Sequence[Hashable],
    axis: int,
    errors: Literal["ignore", "raise"],
) -> pd.DataFrame:
    return df.drop(cols, axis=axis, errors=errors)
