from pydantic import BaseModel, ConfigDict


class ProjectBaseModel(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # allow for any types
        validate_assignment=True,  # run validators when assigning values
        validate_default=True,  # validate default values
        # extra="forbid",  # ensures we don't mis-parse values
        extra="ignore",  # pydantic-cereal `0.0.6` does not support forbidden extra
    )
