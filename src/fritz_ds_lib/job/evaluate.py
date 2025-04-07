from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import load_dataframe
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.utils.utils import init_logger


def evaluate(cfg: AppConfig, dataset: DatasetType) -> None:
    logger = init_logger()

    y_pred = load_dataframe(cfg.predictions_path)
    _, y_true = cfg.loader.load(dataset, return_x_y=True)

    y_pred = y_pred[cfg.model_cfg.evaluation.use_columns]
    y_true = y_true[[cfg.model_cfg.col_target]]
    results = {
        metric.__name__: metric(y_true, y_pred)
        for metric in cfg.model_cfg.evaluation.metrics
    }
    logger.info(f"Metrics for model {cfg.model_cfg.name} are {results}")


def evaluate_all(cfg: AppConfig, dataset: DatasetType) -> None:
    for cfg_ in cfg.iter_models:
        evaluate(cfg_, dataset)
