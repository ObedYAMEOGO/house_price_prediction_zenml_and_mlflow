import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    features = features or []

    strategies = {
        "log": LogTransformation(features),
        "standard_scaling": StandardScaling(features),
        "minmax_scaling": MinMaxScaling(features),
        "onehot_encoding": OneHotEncoding(features),
    }

    if strategy not in strategies:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    engineer = FeatureEngineer(strategies[strategy])
    transformed_df = engineer.apply_feature_engineering(df)

    return transformed_df
