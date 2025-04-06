from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import save_model
from fritz_ds_lib.utils.utils import init_logger


def train(cfg: AppConfig) -> None:
    logger = init_logger()

    model_cfg = cfg.model_cfg

    logger.info("Loading data.")
    df, _ = cfg.loader.load("train")
    y_train = df[model_cfg.col_target]

    logger.info("Starting pipeline.")
    pipe = model_cfg.pipeline
    pipe.fit(df, y_train)
    save_model(pipe, cfg.train_model_path)


def train_all(cfg: AppConfig) -> None:
    for cfg_ in cfg.iter_models:
        train(cfg_)
