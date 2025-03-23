import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging format for better readability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for different outlier detection strategies.

    Why is this needed?
    --------------------
    - Different methods exist for detecting outliers (e.g., Z-score, IQR).
    - Using an abstract class allows easy swapping between methods.
    - Enforces implementation of a `detect_outliers` method in subclasses.
    """

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies outliers in a given dataset.

        Parameters:
        -----------
        - df : pd.DataFrame : Input dataset.

        Returns:
        --------
        - A DataFrame with Boolean values (`True` for outliers, `False` otherwise).
        """
        raise NotImplementedError("Subclasses must implement detect_outliers()")


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    """
    Outlier detection using the Z-score method.

    Why is this needed?
    --------------------
    - Useful for normally distributed data.
    - Identifies outliers as data points deviating more than `threshold` standard deviations.

    Parameters:
    -----------
    - threshold : float : Number of standard deviations to consider an outlier (default = 3).
    """

    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")

        # Compute Z-scores for each feature
        z_scores = np.abs((df - df.mean()) / df.std())

        # Identify outliers based on the threshold
        outliers = z_scores > self.threshold
        logging.info(
            f"Outliers detected using Z-score with threshold: {self.threshold}."
        )
        return outliers


class IQROutlierDetection(OutlierDetectionStrategy):
    """
    Outlier detection using the Interquartile Range (IQR) method.

    Why is this needed?
    --------------------
    - More robust to skewed data than Z-score.
    - Defines outliers as values outside 1.5 times the interquartile range.

    Formula:
    --------
    Outliers = Values < Q1 - 1.5 * IQR or > Q3 + 1.5 * IQR
    """

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")

        # Compute Q1, Q3, and IQR
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        # Identify outliers
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outliers detected using the IQR method.")
        return outliers


class OutlierDetector:
    """
    A class to manage outlier detection, handling, and visualization.

    Why is this needed?
    --------------------
    - Allows flexible switching between detection strategies.
    - Provides methods for handling outliers (removal or capping).
    - Enables visualization for better understanding of outlier distribution.
    """

    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initializes the detector with a specific outlier detection strategy.

        Parameters:
        -----------
        - strategy : OutlierDetectionStrategy : The chosen detection method.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """
        Switches the outlier detection strategy.

        Parameters:
        -----------
        - strategy : OutlierDetectionStrategy : New detection method to use.
        """
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers using the currently selected strategy.

        Parameters:
        -----------
        - df : pd.DataFrame : Dataset to analyze.

        Returns:
        --------
        - DataFrame of the same shape with Boolean values indicating outliers.
        """
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove") -> pd.DataFrame:
        """
        Handles outliers by either removing or capping them.

        Parameters:
        -----------
        - df : pd.DataFrame : Input dataset.
        - method : str : Outlier handling method ('remove' or 'cap').

        Returns:
        --------
        - Processed DataFrame with outliers handled.
        """
        outliers = self.detect_outliers(df)

        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[~outliers.any(axis=1)]  # Remove rows with any outlier

        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(
                lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1
            )

        else:
            logging.warning(
                f"Unknown method '{method}'. No outlier handling performed."
            )
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        """
        Visualizes outliers using boxplots.

        Parameters:
        -----------
        - df : pd.DataFrame : Dataset to visualize.
        - features : list : List of feature names to visualize.
        """
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# Main Execution Block
if __name__ == "__main__":
    pass
