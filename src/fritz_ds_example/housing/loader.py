from datetime import datetime

from pandas import DataFrame

from fritz_ds_lib.data_loading.load import OpenMlRawDataLoader


class CustomDataLoader(OpenMlRawDataLoader):
    def _preprocess(self, df: DataFrame) -> DataFrame:
        df["DateSold"] = df.apply(
            lambda x: datetime(year=x.YrSold, month=x.MoSold, day=1), axis=1
        )
        return df
