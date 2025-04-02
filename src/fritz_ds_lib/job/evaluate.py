import sklearn.metrics

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import load_dataframe
from fritz_ds_lib.core.names import DatasetType


def evaluate(cfg: AppConfig, dataset: DatasetType) -> None:
    y_pred = load_dataframe(cfg)[cfg.model_cfg.col_output]
    _, y_true = cfg.loader.load(dataset, return_x_y=True)
    metric = sklearn.metrics.roc_auc_score
    value = metric(y_true, y_pred)
    print(f"Metric is {value}")
