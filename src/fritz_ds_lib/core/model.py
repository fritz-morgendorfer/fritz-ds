from importlib import import_module
from typing import Any, Callable, Optional

from pydantic import SerializeAsAny, SkipValidation, field_validator
from sklearn.pipeline import Pipeline

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.estimator.adapter import AbstractEstimator
from fritz_ds_lib.model_selection.config import CvConfig
from fritz_ds_lib.pipeline.base import PipelineConfig
from fritz_ds_lib.utils.utils import load_from_dict


class ModelConfig(ProjectBaseModel):
    col_target: str
    col_output: str
    col_idx: Optional[list[str]] = []
    estimator: SkipValidation[SerializeAsAny[AbstractEstimator]]
    prep_pipeline: SkipValidation[SerializeAsAny[PipelineConfig]]
    cv: Optional[CvConfig] = None
    evaluation: list[Callable]

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

    @field_validator("evaluation", mode="before")
    @classmethod
    def _validate_metrics(cls, value: list[str]) -> list[Callable]:
        module = import_module("fritz_ds_lib.evaluation.metrics")
        return [getattr(module, m) for m in value]
