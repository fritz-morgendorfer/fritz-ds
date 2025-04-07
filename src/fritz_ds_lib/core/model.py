from importlib import import_module
from typing import Any, Callable, Literal, Optional

from pydantic import SerializeAsAny, SkipValidation, field_validator
from sklearn.pipeline import Pipeline

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.estimator.adapter import AbstractEstimator
from fritz_ds_lib.model_selection.config import CvConfig
from fritz_ds_lib.pipeline.base import PipelineConfig
from fritz_ds_lib.utils.utils import load_from_dict


class EvaluationConfig(ProjectBaseModel):
    metrics: list[Callable]
    use_columns: list[str] = None

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, value: list[str]) -> list[Callable]:
        module = import_module("sklearn.metrics")
        return [getattr(module, m) for m in value]


class ModelConfig(ProjectBaseModel):
    name: str
    col_target: str
    col_output: list[str]
    col_idx: Optional[list[str]] = []
    prep_pipeline: SkipValidation[SerializeAsAny[PipelineConfig]]
    estimator: SkipValidation[SerializeAsAny[AbstractEstimator]]
    predict: Literal["predict", "predict_proba"] = "predict"
    cv: Optional[CvConfig] = None
    evaluation: EvaluationConfig

    @property
    def pipeline(self) -> Pipeline:
        steps = [
            ("preprocessing", self.prep_pipeline.pipeline),
            ("estimator", self.estimator),
        ]
        return Pipeline(steps=steps)

    @field_validator("estimator", "prep_pipeline", mode="before")
    @classmethod
    def _validate_pydantic_model(cls, value: dict | Any) -> Any:
        if isinstance(value, dict):
            return load_from_dict(value)
        return value
