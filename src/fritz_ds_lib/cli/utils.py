from pathlib import Path
from typing import Union

import joblib
import pandas as pd


def load_model(path: Union[str | Path]):
    model = joblib.load(path)
    return model


def save_model(model, path: Union[str | Path]):
    joblib.dump(model, path)


def save_dataframe(df, path: Union[str | Path]):
    df.to_csv(path)


def load_dataframe(path: Union[str | Path]):
    return pd.read_csv(path)
