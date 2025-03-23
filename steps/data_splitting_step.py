from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets using a predefined strategy.

    Args:
        df (pd.DataFrame): The input dataset containing both features and the target variable.
        target_column (str): The name of the column representing the target variable.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_train (pd.DataFrame): Training feature set.
            - X_test (pd.DataFrame): Testing feature set.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Testing target variable.
    """

    # Initialize the DataSplitter with a simple train-test split strategy
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())

    # Apply the splitting strategy on the dataset
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    # Return the split datasets
    return X_train, X_test, y_train, y_test
