from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from fritz_ds_lib.pipeline.base import PipelineConfig


class DoNothingTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.trained_ = True
        return self

    def transform(self, X):
        return X


def do_nothing_pipeline(cfg: PipelineConfig):
    return Pipeline(steps=[("do_nothing", DoNothingTransform())])
