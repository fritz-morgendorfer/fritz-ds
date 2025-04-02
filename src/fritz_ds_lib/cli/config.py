from typing import Any, Literal

from pydantic import BaseModel, SkipValidation, field_validator

from fritz_ds_lib.core.model import ModelConfig
from fritz_ds_lib.data_loading.load import DataLoader
from fritz_ds_lib.utils import load_from_file


class AppConfig(BaseModel):
    loader: SkipValidation[DataLoader]
    model_cfg: SkipValidation[ModelConfig]
    output_folder: str
    cv_model_file_name: str
    trained_model_file_name: str
    pred_df_file_name: str

    def get_model_file_name(self, source: Literal["cv", "train"]):
        if source == "cv":
            return self.cv_model_file_name
        elif source == "train":
            return self.trained_model_file_name
        else:
            raise ValueError("Unknown source.")

    @field_validator("loader", "model_cfg", mode="before")
    @classmethod
    def _validate_pydantic_model(cls, value: str | Any) -> Any:
        if isinstance(value, str):
            return load_from_file(value)
        return value
