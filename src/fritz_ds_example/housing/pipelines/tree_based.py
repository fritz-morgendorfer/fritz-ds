from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fritz_ds_lib.pipeline.base import PipelineConfig


class TreeBasedPipeCfg(PipelineConfig):
    cols_features: list[str]
    cols_categorical: list[str]


def make_pipeline(cfg: TreeBasedPipeCfg):
    encoder = ColumnTransformer(
        [
            (
                "ord_enc",
                OrdinalEncoder(
                    dtype=int,
                    handle_unknown="use_encoded_value",
                    unknown_value=-2,
                ),
                cfg.cols_categorical,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    selector = ColumnTransformer(
        [
            ("selector", "passthrough", cfg.cols_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps = [
        ("encoder", encoder),
        ("selector", selector),
    ]
    pipeline = Pipeline(steps=steps).set_output(transform="pandas")
    return pipeline
