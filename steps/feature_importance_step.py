import pandas as pd
from src.feature_importance import RandomForestFeatureImportance
from zenml import step


@step
def feature_importance_step(X: pd.DataFrame, y: pd.Series) -> pd.Series:

    if X.empty or y.empty:
        raise ValueError("Input data (X or y) cannot be empty.")

    # Initialize and compute feature importance
    importance_strategy = RandomForestFeatureImportance()
    feature_importances = importance_strategy.compute_importance(X, y)

    return feature_importances
