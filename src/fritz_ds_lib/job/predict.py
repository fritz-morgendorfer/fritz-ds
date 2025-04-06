import pandas as pd

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import load_model, save_dataframe
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.utils.utils import init_logger


def predict(cfg: AppConfig, dataset: DatasetType) -> None:
    logger = init_logger()

    model_cfg = cfg.model_cfg
    logger.info("Loading data.")
    df, _ = cfg.loader.load(dataset)
    idx = df.get(model_cfg.col_idx)
    pipe = load_model(cfg.train_model_path)
    logger.info("Starting pipeline.")
    y_pred = pd.DataFrame(
        pipe.predict(df), columns=[model_cfg.col_output], index=df.index
    )
    y_pred = pd.concat([idx, y_pred], axis=1)
    save_dataframe(y_pred, cfg.predictions_path)


def predict_all(cfg: AppConfig, dataset: DatasetType) -> None:
    for cfg_ in cfg.iter_models:
        predict(cfg_, dataset)
