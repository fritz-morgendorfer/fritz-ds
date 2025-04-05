from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fritz_ds_lib.pipeline.base import PipelineConfig


class HousingLinearModelPipeCfg(PipelineConfig):
    cols_features: list[str]
    cols_categorical: list[str]


def make_pipeline(cfg: HousingLinearModelPipeCfg):
    col_numerical = list(set(cfg.cols_features).difference(cfg.cols_categorical))
    selector = ColumnTransformer(
        [
            ("selector", "passthrough", cfg.cols_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    imputer = ColumnTransformer(
        [
            ("mean_imp", SimpleImputer(), col_numerical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaler = ColumnTransformer(
        [
            ("scaler", MinMaxScaler(), col_numerical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    encoder = ColumnTransformer(
        [
            (
                "oh_enc",
                OneHotEncoder(drop="first", sparse_output=False),
                cfg.cols_categorical,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    steps = [
        ("selector", selector),
        ("imputer", imputer),
        ("scaler", scaler),
        ("encoder", encoder),
    ]
    pipeline = Pipeline(steps=steps).set_output(transform="pandas")
    return pipeline
