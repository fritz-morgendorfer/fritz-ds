import pandas as pd

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import load_model, save_dataframe
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.utils import init_logger


def predict(cfg: AppConfig, dataset: DatasetType) -> None:
    logger = init_logger()

    model_cfg = cfg.model_cfg
    logger.info("Loading data.")
    df, _ = cfg.loader.load(dataset)
    idx = df.get(model_cfg.col_idx)
    pipe = load_model(cfg, "train")
    logger.info("Starting pipeline.")
    y_pred = pd.DataFrame(pipe.predict(df), columns=[model_cfg.col_output])
    y_pred = pd.concat([idx, y_pred])
    save_dataframe(y_pred, cfg)
