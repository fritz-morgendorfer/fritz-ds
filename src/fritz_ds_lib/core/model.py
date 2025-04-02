from typing import Any, Optional

from pydantic import SerializeAsAny, SkipValidation, field_validator
from sklearn.pipeline import Pipeline

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.core.pipeline import PipelineConfig
from fritz_ds_lib.estimator.wrapper import AbstractEstimatorWrapper
from fritz_ds_lib.utils import load_from_dict


class ModelConfig(ProjectBaseModel):
    col_target: str
    col_output: str
    col_idx: Optional[list[str]] = []
    estimator: SkipValidation[SerializeAsAny[AbstractEstimatorWrapper]]
    prep_pipeline: SkipValidation[SerializeAsAny[PipelineConfig]]
    cv: Optional[dict[str, any]] = {}

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
