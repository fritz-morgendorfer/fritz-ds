from sklearn.model_selection import GridSearchCV

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.cli.utils import save_model
from fritz_ds_lib.utils.utils import init_logger


def cv(cfg: AppConfig) -> None:

    logger = init_logger()

    model_cfg = cfg.model_cfg

    logger.info("Loading data.")
    df, y = cfg.loader.load("train")
    target = df[model_cfg.col_target]

    grid_search = GridSearchCV(
        estimator=model_cfg.pipeline,
        scoring=model_cfg.cv.scoring,
        param_grid=model_cfg.cv.param_grid,
        cv=model_cfg.cv.spliter,
        n_jobs=-1,
    )
    logger.info("Starting cross validation.")
    grid_search.fit(df, target)
    save_model(grid_search.best_estimator_, cfg, cfg.cv_model_file_name)
