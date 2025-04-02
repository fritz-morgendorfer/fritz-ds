from pandas import DataFrame


def change_types(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
    for key in mapping:
        df[key] = df[key].astype(mapping[key])
    return df
