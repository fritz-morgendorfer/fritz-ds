from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import load_dataframe
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.utils.utils import init_logger


def evaluate(cfg: AppConfig, dataset: DatasetType) -> None:
    logger = init_logger()

    y_pred = load_dataframe(cfg)
    _, y_true = cfg.loader.load(dataset, return_x_y=True)

    y_pred = y_pred[cfg.model_cfg.col_output]
    y_true = y_true[cfg.model_cfg.col_target]

    results = {
        metric.__name__: metric(y_true, y_pred) for metric in cfg.model_cfg.evaluation
    }
    logger.info(f"Metrics are {results}")
