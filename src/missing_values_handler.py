import logging
from abc import ABC, abstractmethod
import pandas as pd

# Configure logging to track missing value handling steps
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract Base Class for Handling NaN Values
# --------------------------------------------
# Defines a blueprint for different strategies to handle missing values.
# Any subclass must implement the `handle` method.
class NaNValuesHandlingStrategy(ABC):
    """Abstract base class for handling missing values in a DataFrame."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies a missing value handling strategy to a DataFrame."""
        raise NotImplementedError("This method must be implemented in a subclass.")


# Concrete Strategy: Drop NaN Values
# -----------------------------------
# When to Use:
# - Use this when you want to remove rows or columns with missing values.
# - Ideal if missing values are few and dropping them won’t distort analysis.
class DropNaNValuesStrategy(NaNValuesHandlingStrategy):
    """Strategy to drop rows or columns with NaN values."""

    def __init__(self, axis: int = 0, thresh: int = None):
        """
        :param axis:
            - 0 (default) to drop rows containing NaN values.
            - 1 to drop columns containing NaN values.
        :param thresh: Minimum number of non-NaN values required to keep a row/column.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Dropping missing values with axis={self.axis} and thresh={self.thresh}"
        )

        # Drops rows/columns based on the given parameters
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)

        logging.info("Missing values have been dropped.")
        return df_cleaned


# Concrete Strategy: Fill NaN Values
# -----------------------------------
# When to Use:
# - Use this when missing values need to be replaced rather than removed.
# - Various methods available: Mean, Median, Mode, or a Constant Value.
class FillNaNValuesStrategy(NaNValuesHandlingStrategy):
    """Strategy to fill NaN values using different methods."""

    def __init__(self, method: str = "mean", fill_value=None):
        """
        :param method: Strategy to fill missing values, options:
            - "mean": Replace NaN values with the mean of the column.
            - "median": Replace NaN values with the median of the column.
            - "mode": Replace NaN values with the most frequent value.
            - "constant": Replace NaN values with a user-defined constant.
        :param fill_value: Value used when method is "constant".
        """
        self.method = method.lower()
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")
        df_cleaned = df.copy()

        if self.method in ["mean", "median"]:
            # Apply mean/median only to numerical columns
            numeric_columns = df_cleaned.select_dtypes(include="number").columns

            if self.method == "mean":
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                    df[numeric_columns].mean()
                )
            else:
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                    df[numeric_columns].median()
                )

        elif self.method == "mode":
            # Fill each column with its most frequent value
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)

        elif self.method == "constant":
            # Fill NaN values with a specified constant value
            df_cleaned.fillna(self.fill_value, inplace=True)

        else:
            logging.warning(
                f"Unknown method '{self.method}'. No missing values handled."
            )

        logging.info("Missing values have been filled.")
        return df_cleaned


# Context Class: NaN Value Handler
# ---------------------------------
# This class allows dynamic selection of different NaN handling strategies.
# It applies the chosen strategy to a dataset.
class NaNValueHandler:
    """Handles missing values using a specified strategy."""

    def __init__(self, strategy: NaNValuesHandlingStrategy):
        """
        :param strategy: The initial missing value handling strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: NaNValuesHandlingStrategy):
        """Allows changing the missing value handling strategy at runtime."""
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the selected missing value handling strategy to the dataset."""
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)

# Main Execution Block
if __name__ == "__main__":
    pass
