import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Configure logging to display useful information during transformations
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract Base Class for Feature Engineering Strategies
# ------------------------------------------------------
# This defines a blueprint for feature transformation strategies.
# Any subclass must implement the `apply_transformation` method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies a transformation to the given DataFrame."""
        raise NotImplementedError("This method must be implemented in a subclass.")


# Concrete Strategy: Log Transformation
# --------------------------------------
# When to Use:
# - Use log transformation when dealing with **skewed numerical data**.
# - Helps normalize distributions (e.g., income, sales, or population data).
# - Handles `log(0)` issues by using `log1p(x)`, which computes `log(1 + x)`.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """
        :param features: List of numerical features to apply log transformation.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()

        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log(1+x) to handle zeros safely

        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy: Standard Scaling
# -----------------------------------
# When to Use:
# - Use **Standard Scaling (Z-score normalization)** when features have **different units**.
# - It transforms data to have **mean = 0** and **standard deviation = 1**.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """
        :param features: List of numerical features to apply standard scaling.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()

        # Standardizes each feature to mean=0 and std=1
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])

        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy: Min-Max Scaling
# ----------------------------------
# When to Use:
# - Use **Min-Max Scaling** when data should be **bounded** (e.g., scaling values between 0 and 1).
# - Often used in **neural networks** since activation functions expect small values.
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list, feature_range: tuple = (0, 1)):
        """
        :param features: List of numerical features to scale.
        :param feature_range: The target range for scaling (default: 0 to 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()

        # Scales the data between the given feature range
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])

        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy: One-Hot Encoding
# -----------------------------------
# When to Use:
# - Use One-Hot Encoding when working with **categorical features**.
# - Converts categories into **binary (0s and 1s)**, making data machine-learning friendly.
# - Drops the first category (`drop="first"`) to avoid **dummy variable trap**.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """
        :param features: List of categorical features to encode.
        """
        self.features = features
        self.encoder = OneHotEncoder(
            sparse_output=False, drop="first"
        )  # drop="first" to avoid redundancy

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()

        # Apply one-hot encoding and create a new DataFrame with new column names
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )

        # Drop original categorical columns and concatenate the one-hot encoded data
        df_transformed = df_transformed.drop(columns=self.features).reset_index(
            drop=True
        )
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class: Feature Engineer
# -------------------------------
# This class allows dynamic selection of different transformation strategies.
# It applies the chosen transformation to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        :param strategy: The initial feature engineering strategy to apply.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """Allows changing the feature engineering strategy at runtime."""
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the selected feature transformation strategy to the dataset."""
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Main Execution Block
if __name__ == "__main__":
    pass
