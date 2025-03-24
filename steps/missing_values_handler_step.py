import pandas as pd
from src.missing_values_handler import (
    FillNaNValuesStrategy,
    DropNaNValuesStrategy,
    NaNValueHandler,
)
from zenml import step


@step
def missing_values_handler_step(
    df: pd.DataFrame, strategy: str = "mean"
) -> pd.DataFrame:
    strategies = {
        "drop": DropNaNValuesStrategy(axis=0),
        "mean": FillNaNValuesStrategy(method="mean"),
        "median": FillNaNValuesStrategy(method="median"),
        "mode": FillNaNValuesStrategy(method="mode"),
        "constant": FillNaNValuesStrategy(method="constant"),
    }

    if strategy not in strategies:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    handler = NaNValueHandler(strategies[strategy])
    cleaned_df = handler.handle_NaN_values(df)
    return cleaned_df
