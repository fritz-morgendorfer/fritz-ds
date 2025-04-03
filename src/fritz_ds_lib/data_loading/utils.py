import pandas as pd
from pandas import DataFrame


def change_types(
    df: DataFrame, mapping: dict[str, str], parse_dates: list[str]
) -> DataFrame:
    for col in mapping:
        df[col] = df[col].astype(mapping[col])
    for col in parse_dates:
        df[col] = pd.to_datetime(df[col])
    return df


def rename_columns(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
    df = df.rename(columns=mapping)
    return df


def edit_column_names(df: DataFrame) -> DataFrame:
    mapping = {col: col.replace(" ", "_").replace("/", "_") for col in df.columns}
    return rename_columns(df, mapping)
