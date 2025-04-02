from abc import abstractmethod
from typing import Annotated, Optional

from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from fritz_ds_lib.core.names import DatasetType


class AbstractTrainTestSpliter(BaseModel):

    train_size: Annotated[float, Field(gt=0.0, lt=1.0)]
    validation_size: Annotated[float, Field(gt=0.0, lt=1.0)] = None
    test_size: Annotated[float, Field(gt=0.0, lt=1.0)] = None

    @abstractmethod
    def split(self, df: DataFrame, dataset_type: DatasetType) -> DataFrame:
        pass


class SklearnTrainTestSpliter(AbstractTrainTestSpliter):

    random_state: int
    shuffle: bool = True
    stratify: Optional[str] = None  # col name in X dataframe

    def split(self, df: DataFrame, dataset_type: DatasetType) -> DataFrame:
        train_valid_size = self.train_size + self.validation_size

        arrays = [df] if self.stratify is None else [df, df[self.stratify]]
        if self.test_size is not None:
            X, X_test, *rest = train_test_split(
                *arrays,
                train_size=train_valid_size,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=df.get(self.stratify),
            )
        else:
            X, rest = df, [df.get(self.stratify)]

        if self.validation_size is not None:
            X_train, X_validation = train_test_split(
                X,
                train_size=self.train_size / train_valid_size,
                test_size=self.validation_size / train_valid_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=rest[0] if self.stratify is not None else None,
            )
        else:
            X_train = X
        return locals()[f"X_{dataset_type}"]
