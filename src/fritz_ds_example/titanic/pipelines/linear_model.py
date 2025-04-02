from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fritz_ds_lib.core.pipeline import PipelineConfig


class TitanicLinearModelPipeCfg(PipelineConfig):
    cols_used: list[str]


def make_pipeline(cfg: TitanicLinearModelPipeCfg):
    selector = ColumnTransformer(
        [
            ("selector", "passthrough", cfg.cols_used),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    imputer = ColumnTransformer(
        [
            ("mean_imp", SimpleImputer(), ["age"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    encoder = ColumnTransformer(
        [
            (
                "oh_enc",
                OneHotEncoder(drop="first", sparse_output=False),
                ["pclass", "sex"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    steps = [
        ("selector", selector),
        ("imputer", imputer),
        ("encoder", encoder),
    ]
    pipeline = Pipeline(steps=steps).set_output(transform="pandas")
    return pipeline
