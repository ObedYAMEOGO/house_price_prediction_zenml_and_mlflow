import logging
import pandas as pd
from src.outliers_handler import ZScoreOutlierDetection, OutlierDetector
from zenml import step


@step
def outlier_handling_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    return df_cleaned
