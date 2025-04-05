import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fritz_ds_lib.cli.config import AppConfig


def _prepare_folder_for_saving(cfg: AppConfig):
    folder = Path(cfg.output_folder) / cfg.model_cfg.col_target
    os.makedirs(folder, exist_ok=True)
    return folder


def load_model(cfg, source: Literal["cv", "train"]):
    folder = Path(cfg.output_folder) / cfg.model_cfg.col_target
    model = joblib.load(folder / cfg.get_model_file_name(source))
    return model


def save_model(model, cfg, filename):
    folder = _prepare_folder_for_saving(cfg)
    joblib.dump(model, folder / filename)


def save_dataframe(df, cfg):
    folder = _prepare_folder_for_saving(cfg)
    df.to_csv(folder / cfg.pred_df_file_name)


def load_dataframe(cfg):
    folder = Path(cfg.output_folder) / cfg.model_cfg.col_target
    path = folder / cfg.pred_df_file_name
    return pd.read_csv(path)
